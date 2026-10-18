"""A small bag-of-words vocabulary over ORB descriptors, used by _LoopClosureDetector (slam.py) to
shortlist loop-closure candidates without brute-force matching against every earlier keyframe.

The vocabulary (centroids/idf) is a fixed asset trained offline (train_orb_vocabulary) on a
different sequence than it is used on, like DBoW2's in ORB-SLAM, so using it adds no lookahead.
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np

DEFAULT_VOCAB_PATH = Path(__file__).parent / "assets" / "orb_vocab.npz"

# ORB descriptors are 32-byte binary strings compared by Hamming distance (popcount of XOR); this
# table turns a byte popcount into an array lookup.
_POPCOUNT_LUT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def _hamming_distance_matrix(descs: np.ndarray, centroids: np.ndarray, chunk_size: int = 4000) -> np.ndarray:
    """(N, 32) uint8 x (K, 32) uint8 -> (N, K) int32 pairwise Hamming distances.

    Chunked over N so a large batch doesn't blow up the (N, K, 32) XOR intermediate (~128MB at
    chunk_size=4000, K=1000).
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
        """A sparse, L2-normalized TF-IDF bag-of-words vector: word id -> weight. The dot product of
        two keyframes' vectors is a cheap appearance-similarity score.
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
    """K-means over binary ORB descriptors: nearest-centroid assignment by Hamming distance, new
    centroid = per-bit majority vote of its members (as in DBoW2). A cluster that loses every member
    keeps its previous centroid.

    descriptor_sets is one array per training "document" (keyframe), kept separate because IDF needs
    per-document presence, not global counts.
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

    # IDF from the same corpus, one document per keyframe, smoothed so a word absent from
    # training never gets a zero/undefined weight.
    doc_count = np.zeros(k, dtype=np.int64)
    for descs in descriptor_sets:
        if descs is None or len(descs) == 0:
            continue
        present = np.unique(_hamming_distance_matrix(descs, centroids).argmin(axis=1))
        doc_count[present] += 1
    idf = np.log((len(descriptor_sets) + 1) / (doc_count + 1)) + 1.0

    return ORBVocabulary(centroids=centroids, idf=idf)
