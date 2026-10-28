from dataclasses import dataclass

import cv2
import numpy as np

from slam.data import EuRoCMAVData


# Reject a stereo match whose undistorted points sit more than this far [px] (Sampson distance)
# from the epipolar geometry of the known calibration. Tight because F is exact: derived from the
# fixed extrinsics, not estimated from noisy matches.
EPIPOLAR_SAMPSON_PX = 1.5
# Candidate band for matching: cam1 keypoints farther than this [px] from a cam0 keypoint's
# epipolar line aren't compared at all. Looser than the Sampson gate (a pair passing the gate can
# be up to ~1.4x its Sampson distance from the line) so the gate stays the real filter.
EPIPOLAR_CANDIDATE_PX = 3.0


def _skew(t: np.ndarray) -> np.ndarray:
    return np.array([
        [0.0, -t[2], t[1]],
        [t[2], 0.0, -t[0]],
        [-t[1], t[0], 0.0],
    ])


class StereoGeometry:
    """The rig's fixed stereo geometry, built once from the calibration: undistortion, the exact
    fundamental matrix, the Sampson epipolar distance, and triangulation. Shared by the keyframe ORB
    matcher below and the per-frame KLT stereo in optical_flow.py.
    """

    def __init__(self, data: EuRoCMAVData) -> None:
        self.K0 = data.cam0_intrinsics.to_matrix()
        self.K1 = data.cam1_intrinsics.to_matrix()
        self.dist0 = np.array([
            data.cam0_intrinsics.k1, data.cam0_intrinsics.k2, data.cam0_intrinsics.p1, data.cam0_intrinsics.p2])
        self.dist1 = np.array([
            data.cam1_intrinsics.k1, data.cam1_intrinsics.k2, data.cam1_intrinsics.p1, data.cam1_intrinsics.p2])
        T_cam0_to_cam1 = np.linalg.inv(data.cam1_extrinsics) @ data.cam0_extrinsics
        R = T_cam0_to_cam1[:3, :3]
        t = T_cam0_to_cam1[:3, 3]
        self.F = np.linalg.inv(self.K1).T @ _skew(t) @ R @ np.linalg.inv(self.K0)
        self.P0 = self.K0 @ np.hstack([np.eye(3), np.zeros((3, 1))])
        self.P1 = self.K1 @ T_cam0_to_cam1[:3, :]

    def undistort0(self, pts: np.ndarray) -> np.ndarray:
        """(N, 2) distorted cam0 pixels -> (N, 2) undistorted pixels."""
        return cv2.undistortPoints(
            pts.reshape(-1, 1, 2).astype(np.float64), self.K0, self.dist0, P=self.K0).reshape(-1, 2)

    def undistort1(self, pts: np.ndarray) -> np.ndarray:
        return cv2.undistortPoints(
            pts.reshape(-1, 1, 2).astype(np.float64), self.K1, self.dist1, P=self.K1).reshape(-1, 2)

    def sampson_sq(self, points0: np.ndarray, points1: np.ndarray) -> np.ndarray:
        """Squared Sampson distance [px^2] of each undistorted (cam0, cam1) pair from the epipolar
        geometry: a correct match satisfies x1^T F x0 = 0.
        """
        p0h = np.hstack([points0, np.ones((len(points0), 1))])
        p1h = np.hstack([points1, np.ones((len(points1), 1))])
        Fp0 = p0h @ self.F.T      # epipolar line in cam1 for each cam0 point
        Ftp1 = p1h @ self.F       # epipolar line in cam0 for each cam1 point
        num = np.sum(p1h * Fp0, axis=1)
        denom = Fp0[:, 0] ** 2 + Fp0[:, 1] ** 2 + Ftp1[:, 0] ** 2 + Ftp1[:, 1] ** 2
        return np.where(denom > 1e-12, num ** 2 / np.maximum(denom, 1e-12), np.inf)

    def triangulate(self, points0: np.ndarray, points1: np.ndarray) -> np.ndarray:
        """Undistorted (N, 2) pairs -> (3, N) points in the cam0 frame."""
        points_4d = cv2.triangulatePoints(self.P0, self.P1, points0.T, points1.T)
        return points_4d[:3, :] / points_4d[3, :]


@dataclass
class StereoMatchingFrame:
    timestamp_ns: int
    matches: list  # list[cv2.DMatch]
    points_3d: np.ndarray  # shape (3, N)


def match_and_triangulate_stereo(
    geometry: StereoGeometry,
    cam0_keypoints: list, cam0_descriptors: np.ndarray,
    cam1_keypoints: list, cam1_descriptors: np.ndarray,
) -> tuple[list, np.ndarray]:
    """Ratio-test + epipolar-gated cam0<->cam1 ORB match, then triangulate.

    Returns (matches, points_3d) with points_3d shape (3, N), points_3d[:, i] <-> matches[i]
    (queryIdx into cam0_keypoints/descriptors, trainIdx into cam1's).
    """
    if len(cam0_keypoints) == 0 or len(cam1_keypoints) == 0:
        return [], np.zeros((3, 0))

    all_points0 = geometry.undistort0(cv2.KeyPoint_convert(cam0_keypoints))
    all_points1 = geometry.undistort1(cv2.KeyPoint_convert(cam1_keypoints))

    # Only cam1 keypoints within EPIPOLAR_CANDIDATE_PX of a cam0 keypoint's epipolar line are
    # considered as its match: BFMatcher skips masked pairs, so this avoids almost the whole
    # N x M descriptor comparison. The ratio test below then runs among the geometrically
    # plausible candidates only.
    h0 = np.hstack([all_points0, np.ones((len(all_points0), 1))])
    h1 = np.hstack([all_points1, np.ones((len(all_points1), 1))])
    lines = h0 @ geometry.F.T  # epipolar line in cam1 for each cam0 keypoint
    lines /= np.maximum(np.linalg.norm(lines[:, :2], axis=1, keepdims=True), 1e-12)
    line_dist = (lines.astype(np.float32) @ h1.astype(np.float32).T)  # signed distance, (N0, N1)
    mask = cv2.inRange(line_dist, -EPIPOLAR_CANDIDATE_PX, EPIPOLAR_CANDIDATE_PX)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    raw_matches = bf.knnMatch(cam0_descriptors, cam1_descriptors, k=2, mask=mask)
    # A keypoint with fewer than two candidates in its band can't be ratio-tested; drop it.
    good_matches = [
        pair[0] for pair in raw_matches if len(pair) == 2 and pair[0].distance < 0.75 * pair[1].distance]
    if not good_matches:
        return [], np.zeros((3, 0))

    points0 = all_points0[[m.queryIdx for m in good_matches]]
    points1 = all_points1[[m.trainIdx for m in good_matches]]

    # Drop repeated-texture mismatches before they triangulate into garbage.
    keep = geometry.sampson_sq(points0, points1) < EPIPOLAR_SAMPSON_PX ** 2
    good_matches = [m for m, k in zip(good_matches, keep) if k]
    if not good_matches:
        return [], np.zeros((3, 0))

    return good_matches, geometry.triangulate(points0[keep], points1[keep])
