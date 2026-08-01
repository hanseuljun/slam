"""A small bag-of-words vocabulary over ORB descriptors, used by _LoopClosureDetector
(slam.py) to shortlist loop-closure candidates without a brute-force descriptor-to-descriptor
probe against every earlier keyframe -- see that class's docstring for why.

The vocabulary itself (ORBVocabulary.centroids/idf) is a fixed, offline-trained asset, not
something derived from a live run's own frames: it's trained once (train_orb_vocabulary, driven
by tmp/investigate/train_orb_vocab.py) from a *different* sequence than whatever it's later used
to detect closures on, exactly like DBoW2's vocabulary in ORB-SLAM. That's what keeps it from
reintroducing the kind of look-ahead the rest of this module has been built to avoid: nothing
about using the vocabulary at query time depends on any keyframe not yet added.
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np

DEFAULT_VOCAB_PATH = Path(__file__).parent / "assets" / "orb_vocab.npz"

# ORB descriptors are 32-byte (256-bit) binary strings; distance between them is Hamming
# distance (popcount of XOR). This 256-entry table turns "popcount of a byte" into an array
# lookup instead of a per-byte Python loop.
_POPCOUNT_LUT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def _hamming_distance_matrix(descs: np.ndarray, centroids: np.ndarray, chunk_size: int = 4000) -> np.ndarray:
    """(N, 32) uint8 x (K, 32) uint8 -> (N, K) int32 pairwise Hamming distances.

    Chunked over N (not K -- vocabularies here are at most a few thousand words, small enough to
    broadcast against whole) so a large descriptor batch doesn't blow up the (N, K, 32) XOR
    intermediate: at chunk_size=4000 and K=1000, that intermediate is ~128MB, not gigabytes.
    """
    n = descs.shape[0]
    k = centroids.shape[0]
    out = np.empty((n, k), dtype=np.int32)
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        xor = descs[start:end, None, :] ^ centroids[None, :, :]
        out[start:end] = _POPCOUNT_LUT[xor].sum(axis=2)
    return out


@dataclass
class ORBVocabulary:
    centroids: np.ndarray  # (K, 32) uint8 -- one binary "visual word" per row
    idf: np.ndarray        # (K,) float64 -- inverse document frequency, from training corpus

    @property
    def size(self) -> int:
        return len(self.centroids)

    def nearest_words(self, descriptors: np.ndarray) -> np.ndarray:
        """(N, 32) uint8 -> (N,) int, each descriptor's nearest-centroid word id."""
        return _hamming_distance_matrix(descriptors, self.centroids).argmin(axis=1)

    def transform(self, descriptors: np.ndarray) -> dict[int, float]:
        """A sparse, L2-normalized TF-IDF bag-of-words vector: word id -> weight. Two keyframes'
        vectors' dot product is a cheap appearance-similarity score -- no descriptor-to-descriptor
        matching needed to compare them.
        """
        if descriptors is None or len(descriptors) == 0:
            return {}
        words = self.nearest_words(descriptors)
        counts = np.bincount(words, minlength=self.size).astype(np.float64)
        weights = (counts / counts.sum()) * self.idf
        norm = np.linalg.norm(weights)
        if norm < 1e-12:
            return {}
        weights /= norm
        nz = np.nonzero(weights)[0]
        return {int(w): float(weights[w]) for w in nz}

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, centroids=self.centroids, idf=self.idf)

    @classmethod
    def load(cls, path: Path = DEFAULT_VOCAB_PATH) -> "ORBVocabulary":
        data = np.load(path)
        return cls(centroids=data["centroids"], idf=data["idf"])


def train_orb_vocabulary(
    descriptor_sets: list[np.ndarray], k: int, iterations: int = 8, seed: int = 0,
) -> ORBVocabulary:
    """K-means over binary ORB descriptors: nearest-centroid assignment is by Hamming distance,
    and a cluster's new centroid is the per-bit majority vote of its assigned descriptors (the
    Hamming-space analogue of an arithmetic mean) -- standard for binary descriptor vocabularies
    (DBoW2 uses the same idea). A cluster that loses every member on some iteration keeps its
    previous centroid rather than being reseeded, to keep this simple; with k small relative to
    a large, diverse training corpus this is rare in practice.

    descriptor_sets is one array of descriptors per training "document" (keyframe) -- kept
    separate, not pre-concatenated, because IDF needs per-document presence/absence, not a global
    descriptor count.
    """
    rng = np.random.default_rng(seed)
    all_descs = np.concatenate(descriptor_sets, axis=0)
    n = len(all_descs)
    centroids = all_descs[rng.choice(n, size=k, replace=False)].copy()

    for _ in range(iterations):
        assignments = _hamming_distance_matrix(all_descs, centroids).argmin(axis=1)
        bits = np.unpackbits(all_descs, axis=1)  # (n, 256)
        new_centroids = centroids.copy()
        counts = np.bincount(assignments, minlength=k)
        for word in np.nonzero(counts)[0]:
            mask = assignments == word
            majority_bits = (bits[mask].mean(axis=0) >= 0.5).astype(np.uint8)
            new_centroids[word] = np.packbits(majority_bits)
        centroids = new_centroids

    # IDF from the same corpus, one "document" per keyframe's descriptor set: how many keyframes
    # contain each word at least once, smoothed so a word absent from training never gets a
    # zero/undefined weight at query time.
    doc_count = np.zeros(k, dtype=np.int64)
    for descs in descriptor_sets:
        if descs is None or len(descs) == 0:
            continue
        present = np.unique(_hamming_distance_matrix(descs, centroids).argmin(axis=1))
        doc_count[present] += 1
    idf = np.log((len(descriptor_sets) + 1) / (doc_count + 1)) + 1.0

    return ORBVocabulary(centroids=centroids, idf=idf)
