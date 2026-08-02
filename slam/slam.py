from collections import defaultdict
from dataclasses import dataclass
import threading
import time
import traceback
from typing import Callable, Optional

import cv2
import gtsam
import numpy as np

from slam.data import EuRoCMAVData, ImuSample
from slam.feature_detection import FeatureDetectionFrame, FeatureDetectionResult, detect_features_for_frame
from slam.imu_initialization import ImuInitializationResult
from slam.optical_flow import OpticalFlowFrame, OpticalFlowResult, OpticalFlowTracker
from slam.orb_vocabulary import ORBVocabulary
from slam.stereo_matching import StereoMatchingFrame, StereoMatchingResult, match_and_triangulate_stereo
from slam.util import quaternion_to_rotation_matrix

RPE_DELTA_S = 1.0  # RPE window: how far apart (in time) the two poses being compared are [s]


class _FrontendFrameComputer:
    """Feature detection + stereo matching + optical flow for one frame at a time -- the per-frame
    body _compute_frontend_incrementally (below) loops over, pulled out so _compute's own
    per-frame loop (which also drives _KeyframeScanner/_GtsamBuilder) can call it directly instead
    of needing a separate complete pass before keyframe selection can even start.

    Each camera image is loaded once and reused for both feature detection and optical flow
    (detect_features_for_frame accepts pre-loaded images for exactly this reason) -- today's
    separate FeatureDetectionSolver/OpticalFlowSolver each load cam0 independently. A single
    OpticalFlowTracker is owned across the whole lifetime of this object, since track identity has
    to stay continuous frame to frame the same way it does inside OpticalFlowSolver.run() -- see
    OpticalFlowTracker's docstring.
    """

    def __init__(self, data: EuRoCMAVData) -> None:
        self._data = data
        self._tracker = OpticalFlowTracker()

    def add_frame(self, ts: int) -> tuple[FeatureDetectionFrame, StereoMatchingFrame, OpticalFlowFrame]:
        cam0_img = cv2.imread(str(self._data.get_cam0_image_path(ts)), cv2.IMREAD_GRAYSCALE)
        cam1_img = cv2.imread(str(self._data.get_cam1_image_path(ts)), cv2.IMREAD_GRAYSCALE)
        fd_frame = detect_features_for_frame(self._data, ts, cam0_img, cam1_img)

        matches, points_3d = match_and_triangulate_stereo(
            self._data, fd_frame.cam0_keypoints, fd_frame.cam0_descriptors,
            fd_frame.cam1_keypoints, fd_frame.cam1_descriptors)
        sm_frame = StereoMatchingFrame(timestamp_ns=ts, matches=matches, points_3d=points_3d)

        of_frame = self._tracker.add_frame(fd_frame, sm_frame, cam0_img)
        return fd_frame, sm_frame, of_frame


def _compute_frontend_incrementally(
    data: EuRoCMAVData,
    start_s: float,
    duration_s: float,
    cancel_event: threading.Event,
    on_progress: Optional[Callable[[float], None]] = None,
) -> tuple[FeatureDetectionResult, StereoMatchingResult, OpticalFlowResult]:
    """Feature detection + stereo matching + optical flow for every frame in
    [start_s, start_s + duration_s], computed one frame at a time in strictly increasing order via
    _FrontendFrameComputer -- no ThreadPoolExecutor precomputation across the window (that's what
    FeatureDetectionSolver/StereoMatchingSolver do; both are safe to call one frame at a time
    since neither carries state across frames), and no frame ever read before its own turn.

    Produces the exact same FeatureDetectionResult/StereoMatchingResult/OpticalFlowResult shape
    the batch solvers return -- this only changes *how* that data gets computed (incrementally,
    inside this module, from raw images) rather than *what* it contains.
    """
    first_ts = data.cam_timestamps_ns[0]
    min_ts = first_ts + int(start_s * 1e9)
    max_ts = min_ts + int(duration_s * 1e9)
    timestamps = [t for t in data.cam_timestamps_ns if min_ts <= t <= max_ts]
    n = len(timestamps)

    computer = _FrontendFrameComputer(data)
    fd_frames: list[FeatureDetectionFrame] = []
    sm_frames: list[StereoMatchingFrame] = []
    of_frames: list[OpticalFlowFrame] = []

    t0 = time.monotonic()
    for i, ts in enumerate(timestamps):
        if cancel_event.is_set():
            break
        fd_frame, sm_frame, of_frame = computer.add_frame(ts)
        fd_frames.append(fd_frame)
        sm_frames.append(sm_frame)
        of_frames.append(of_frame)
        if on_progress is not None:
            on_progress((i + 1) / n)

    elapsed_s = time.monotonic() - t0
    return (
        FeatureDetectionResult(frames=fd_frames, elapsed_s=elapsed_s),
        StereoMatchingResult(frames=sm_frames, elapsed_s=elapsed_s),
        OpticalFlowResult(frames=of_frames, elapsed_s=elapsed_s),
    )


@dataclass
class SlamGroundTruthResult:
    times: np.ndarray
    positions: np.ndarray
    attitudes: np.ndarray
    rotation_matrices: np.ndarray
    angular_velocity_times: np.ndarray
    angular_velocities: np.ndarray


@dataclass
class SlamImuResult:
    times: np.ndarray
    attitudes: np.ndarray
    rotation_matrices: np.ndarray
    angular_velocities: np.ndarray
    linear_accelerations: np.ndarray


@dataclass
class SlamPnpResult:
    times: np.ndarray
    positions: np.ndarray
    attitudes: np.ndarray
    rotation_matrices: np.ndarray
    angular_velocity_times: np.ndarray
    angular_velocities: np.ndarray
    elapsed_time: float = 0.0


@dataclass
class SlamGtsamResult:
    times: np.ndarray
    positions: np.ndarray
    attitudes: np.ndarray
    rotation_matrices: np.ndarray
    velocities: np.ndarray
    biases: np.ndarray  # per-keyframe IMU bias, shape (K, 6): [accel(3), gyro(3)]
    position_errors: np.ndarray  # per-keyframe position error vs nearest GT sample [m], shape (K,)
    # ATE: batch (whole-trajectory) yaw+translation alignment, then per-keyframe error. Yaw-only
    # (not full 3D rotation) because gravity makes roll/pitch observable for a VIO system; letting
    # the alignment absorb them would hide real error instead of exposing it. See Zhang &
    # Scaramuzza, "A Tutorial on Quantitative Trajectory Evaluation for VIO", IROS 2018.
    ate_position_errors: np.ndarray  # shape (K,) [m]
    ate_rotation_errors: np.ndarray  # shape (K,) [deg]
    # RPE: relative motion error over a fixed time window, alignment-free by construction (a
    # constant misalignment cancels out of a relative transform), so it isolates local/per-step
    # accuracy from the cumulative drift ATE reports. NaN where no keyframe falls within the
    # window of the trajectory's end.
    rpe_translation_errors: np.ndarray  # shape (K,) [m]
    rpe_rotation_errors: np.ndarray  # shape (K,) [deg]
    reprojection_rmse: np.ndarray  # per-keyframe landmark reprojection RMSE [px], shape (K,)
    landmark_counts: np.ndarray  # per-keyframe number of observed landmarks, shape (K,)
    angular_velocity_times: np.ndarray
    angular_velocities: np.ndarray
    linear_accelerations: np.ndarray
    elapsed_time: float = 0.0


@dataclass
class SlamExtraResult:
    gravity: np.ndarray
    linear_accelerations_in_world: np.ndarray


@dataclass
class SlamResult:
    gt: SlamGroundTruthResult
    imu: SlamImuResult
    pnp: SlamPnpResult
    gtsam: SlamGtsamResult
    extra: SlamExtraResult



def _mats_to_rvecs(rotation_matrices: np.ndarray) -> np.ndarray:
    return np.array([cv2.Rodrigues(R)[0].flatten() for R in rotation_matrices])


def _rvec_tvec_to_transform(rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3], _ = cv2.Rodrigues(rvec)
    T[:3, 3] = tvec.flatten()
    return T


def _to_gtsam_pose3(T: np.ndarray) -> gtsam.Pose3:
    return gtsam.Pose3(gtsam.Rot3(T[:3, :3]), gtsam.Point3(*T[:3, 3]))


def _elapsed_s(timestamp_ns: int, reference_ns: int) -> float:
    return (timestamp_ns - reference_ns) / 1e9


def _align_vectors(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Rotation matrix R such that R @ a is parallel to b (both unit-normalized)."""
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    if c > 1.0 - 1e-8:
        return np.eye(3)
    if c < -1.0 + 1e-8:
        # a and b are antiparallel; rotate 180 deg about any axis orthogonal to a.
        axis = np.cross(a, np.array([1.0, 0.0, 0.0]))
        if np.linalg.norm(axis) < 1e-6:
            axis = np.cross(a, np.array([0.0, 1.0, 0.0]))
        axis = axis / np.linalg.norm(axis)
        return 2.0 * np.outer(axis, axis) - np.eye(3)
    vx = np.array([
        [0.0, -v[2], v[1]],
        [v[2], 0.0, -v[0]],
        [-v[1], v[0], 0.0],
    ])
    return np.eye(3) + vx + vx @ vx * (1.0 / (1.0 + c))


def _rotation_angle_deg(rotation_matrix: np.ndarray) -> float:
    """Geodesic angle of a rotation matrix (its distance from identity), in degrees."""
    cos_angle = np.clip((np.trace(rotation_matrix) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_angle)))


def _nearest_sorted_indices(sorted_reference: np.ndarray, queries: np.ndarray) -> np.ndarray:
    """For each query, the index into sorted_reference whose value is closest -- same result as
    argmin(abs(sorted_reference[:, None] - queries[None, :]), axis=0), but via binary search
    instead of a dense (len(sorted_reference) x len(queries)) broadcast. That broadcast is O(n*m)
    in both time and memory: matching ~29k IMU samples against ~29k ground-truth samples over a
    145s sequence built an 836M-element array (~6.7GB, plus another same-sized array for the abs())
    for a single nearest-timestamp lookup. This is O((n + m) log n) time, O(m) memory.
    """
    idx = np.searchsorted(sorted_reference, queries, side="left")
    idx = np.clip(idx, 1, len(sorted_reference) - 1)
    left, right = sorted_reference[idx - 1], sorted_reference[idx]
    return idx - ((queries - left) <= (right - queries))


def _yaw_translation_align(est_positions: np.ndarray, gt_positions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Least-squares rotation-about-z (yaw) + translation aligning est_positions to gt_positions.

    Restricted to yaw rather than a full 3D (Umeyama) rotation: both trajectories are already
    gravity-aligned (nav frame's z is "up"), and for a VIO system gravity makes roll/pitch
    observable, so a full-rotation alignment could quietly absorb a real roll/pitch error instead
    of reporting it. Closed form: for fixed yaw the optimal translation is the centroid offset;
    substituting that in reduces the remaining 1D optimization to maximizing
    A*cos(theta) + B*sin(theta), solved by theta = atan2(B, A).

    Only observable when the trajectory has real horizontal spread: fit purely from positions, so
    over a short or near-stationary window (e.g. EuRoC's static lead-in before takeoff) the
    horizontal signal is too small to pin theta down, and the resulting yaw -- and therefore
    ate_rotation_errors -- can be noisy or biased even though the fit is the true least-squares
    optimum for that window. Not a defect in the estimate; a property of evaluating too short/still
    a stretch. Prefer windows with a few meters of horizontal travel when reading ATE rotation.
    """
    p_mean, q_mean = est_positions.mean(axis=0), gt_positions.mean(axis=0)
    p, q = est_positions - p_mean, gt_positions - q_mean
    a = float(np.sum(q[:, 0] * p[:, 0] + q[:, 1] * p[:, 1]))
    b = float(np.sum(q[:, 1] * p[:, 0] - q[:, 0] * p[:, 1]))
    theta = np.arctan2(b, a)
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    t = q_mean - R @ p_mean
    return R, t


def _compute_ate(
    positions: np.ndarray, rotation_matrices: np.ndarray,
    gt_positions: np.ndarray, gt_rotation_matrices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Absolute Trajectory Error: batch-align the whole estimated trajectory to ground truth
    once (yaw + translation), then report per-keyframe position/rotation error after that single
    alignment -- comparable to how other VIO/SLAM systems' ATE is usually reported."""
    R, t = _yaw_translation_align(positions, gt_positions)
    aligned_positions = positions @ R.T + t
    position_errors = np.linalg.norm(aligned_positions - gt_positions, axis=1)
    aligned_rotations = np.einsum('ij,kjl->kil', R, rotation_matrices)
    rotation_errors = np.array([
        _rotation_angle_deg(gt_rotation_matrices[i].T @ aligned_rotations[i])
        for i in range(len(positions))
    ])
    return position_errors, rotation_errors


def _compute_rpe(
    times: np.ndarray, positions: np.ndarray, rotation_matrices: np.ndarray,
    gt_positions: np.ndarray, gt_rotation_matrices: np.ndarray, delta_s: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Relative Pose Error: for each keyframe, compare the estimated relative motion over the
    next delta_s seconds to ground truth's relative motion over the same interval. Needs no
    alignment -- a constant misalignment cancels out of a relative (inv(T_i) @ T_j) transform --
    so unlike ATE this isolates local/per-step accuracy from cumulative drift."""
    K = len(times)
    translation_errors = np.full(K, np.nan)
    rotation_errors = np.full(K, np.nan)
    for i in range(K):
        if times[i] + delta_s > times[-1]:
            continue
        j = int(np.argmin(np.abs(times - (times[i] + delta_s))))
        if j <= i:
            continue
        rel_est = np.linalg.inv(rotation_matrices[i]) @ (positions[j] - positions[i])
        rel_est_R = rotation_matrices[i].T @ rotation_matrices[j]
        rel_gt = gt_rotation_matrices[i].T @ (gt_positions[j] - gt_positions[i])
        rel_gt_R = gt_rotation_matrices[i].T @ gt_rotation_matrices[j]
        translation_errors[i] = float(np.linalg.norm(rel_est - rel_gt))
        rotation_errors[i] = _rotation_angle_deg(rel_gt_R.T @ rel_est_R)
    return translation_errors, rotation_errors


def _get_ground_truth_result(
    data: EuRoCMAVData,
    first_timestamp_ns: int,
    min_timestamp_ns: int,
    max_timestamp_ns: int,
) -> SlamGroundTruthResult:
    samples = [s for s in data.ground_truth_samples if min_timestamp_ns <= s.timestamp_ns <= max_timestamp_ns]
    times = np.array([_elapsed_s(s.timestamp_ns, first_timestamp_ns) for s in samples])
    positions = np.array([s.position for s in samples])
    rotation_matrices = np.array([quaternion_to_rotation_matrix(s.quaternion) for s in samples])

    angular_velocities = []
    for j in range(len(samples) - 1):
        rotation = rotation_matrices[j].T @ rotation_matrices[j + 1]
        rotation_vector, _ = cv2.Rodrigues(rotation)
        dt = _elapsed_s(samples[j + 1].timestamp_ns, samples[j].timestamp_ns)
        angular_velocity = rotation_vector.flatten() / dt
        angular_velocities.append(angular_velocity)
    angular_velocities = np.array(angular_velocities)

    return SlamGroundTruthResult(
        times=times,
        positions=positions,
        attitudes=_mats_to_rvecs(rotation_matrices),
        rotation_matrices=rotation_matrices,
        angular_velocity_times=np.array([_elapsed_s(s.timestamp_ns, first_timestamp_ns) for s in samples[:-1]]),
        angular_velocities=angular_velocities,
    )


def _get_imu_result(
    data: EuRoCMAVData,
    first_timestamp_ns: int,
    min_timestamp_ns: int,
    max_timestamp_ns: int,
    gt_rotation_matrices: np.ndarray,
    first_gt_timestamp_ns: int,
) -> SlamImuResult:
    samples = [s for s in data.imu_samples if min_timestamp_ns <= s.timestamp_ns <= max_timestamp_ns]
    times = np.array([_elapsed_s(s.timestamp_ns, first_timestamp_ns) for s in samples])
    linear_accelerations = np.array([s.linear_acceleration for s in samples])
    angular_velocities = np.array([s.angular_velocity for s in samples])
    rotation_matrices_list = []
    prev_rotation_matrix = np.eye(3)
    for angular_velocity in angular_velocities:
        rotation_matrix, _ = cv2.Rodrigues(angular_velocity / data.imu0_rate_hz)
        prev_rotation_matrix = prev_rotation_matrix @ rotation_matrix
        rotation_matrices_list.append(prev_rotation_matrix)

    timestamps_ns = np.array([s.timestamp_ns for s in samples])
    index_closest_to_first_gt_sample = np.argmin(np.abs(timestamps_ns - first_gt_timestamp_ns))
    compensation_rotation_matrix = gt_rotation_matrices[0] @ rotation_matrices_list[index_closest_to_first_gt_sample].T
    for i in range(len(rotation_matrices_list)):
        rotation_matrices_list[i] = compensation_rotation_matrix @ rotation_matrices_list[i]

    rotation_matrices = np.array(rotation_matrices_list)

    return SlamImuResult(
        times=times,
        attitudes=_mats_to_rvecs(rotation_matrices),
        rotation_matrices=rotation_matrices,
        angular_velocities=angular_velocities,
        linear_accelerations=linear_accelerations,
    )


def _run_pnp_step(
    data: EuRoCMAVData,
    points_3d: np.ndarray,
    stereo_matches: list,
    cam0_descriptors0: np.ndarray,
    cam0_keypoints1,
    cam0_descriptors1: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int, float]:
    cam0_idx_to_3d_idx = {m.queryIdx: i for i, m in enumerate(stereo_matches)}

    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    temporal_matches = bf.knnMatch(cam0_descriptors0, cam0_descriptors1, k=2)
    temporal_good_matches = [m for m, n in temporal_matches if m.distance < 0.75 * n.distance]

    object_points = []
    image_points = []
    for m in temporal_good_matches:
        if m.queryIdx in cam0_idx_to_3d_idx:
            idx_3d = cam0_idx_to_3d_idx[m.queryIdx]
            object_points.append(points_3d[:, idx_3d])
            image_points.append(cam0_keypoints1[m.trainIdx].pt)

    object_points = np.array(object_points, dtype=np.float64)
    image_points = np.array(image_points, dtype=np.float64)

    intrinsics_matrix = data.cam0_intrinsics.to_matrix()
    dist_coeffs = np.array([
        data.cam0_intrinsics.k1, data.cam0_intrinsics.k2,
        data.cam0_intrinsics.p1, data.cam0_intrinsics.p2,
    ])

    success, rotation_vector, translation_vector, inliers = cv2.solvePnPRansac(object_points, image_points, intrinsics_matrix, dist_coeffs)
    if not success:
        raise RuntimeError(f"cv2.solvePnPRansac failed. len(object_points): {len(object_points)}")

    # cv2's stub types `inliers` too loosely for numpy's __getitem__ overloads to match, even
    # though this is ordinary integer-array indexing at runtime.
    inlier_object_points = object_points[inliers.flatten()]  # type: ignore[index]
    inlier_image_points = image_points[inliers.flatten()]  # type: ignore[index]
    projected, _ = cv2.projectPoints(inlier_object_points, rotation_vector, translation_vector, intrinsics_matrix, dist_coeffs)
    reprojection_error = np.mean(np.linalg.norm(inlier_image_points - projected.reshape(-1, 2), axis=1))

    # cv2.solvePnPRansac returns the world<-camera transform; the rest of this module treats
    # poses as camera<-world (a camera's pose expressed in world coordinates), so invert here.
    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
    rotation_vector = -rotation_vector
    translation_vector = -rotation_matrix.T @ translation_vector

    return rotation_vector, translation_vector, len(temporal_good_matches), reprojection_error


@dataclass(frozen=True)
class LoopClosureCandidate:
    from_frame: int              # earlier keyframe (into stereo_matching_result.frames)
    to_frame: int                # later keyframe ("query") that revisits from_frame
    body_relative_pose: np.ndarray  # 4x4, from_frame_body <- to_frame_body
    num_matches: int
    reprojection_error: float


@dataclass(frozen=True)
class LoopClosureFactor:
    """A closure ready to insert into the pose graph -- either auto-detected (converted from a
    LoopClosureCandidate, see _run_gtsam) or supplied directly as a manual test/override hook
    (_run_gtsam's extra_loop_closures). from_frame/to_frame are frame indices, not node indices --
    see extra_loop_closures' docstring for why."""
    from_frame: int
    to_frame: int
    body_relative_pose: np.ndarray  # 4x4, from_frame_body <- to_frame_body
    rot_sigma: float   # rad
    trans_sigma: float  # m
    noise_mode: str    # "gaussian" | "huber" | "dcs" -- see _run_gtsam's noise construction


@dataclass(frozen=True)
class _LoopClosureEdge:
    """LoopClosureFactor (or an auto-detected LoopClosureCandidate) with both endpoints resolved
    to graph node indices."""
    from_node: int
    to_node: int
    body_relative_pose: np.ndarray
    rot_sigma: float
    trans_sigma: float
    noise_mode: str


# Sanity bound on a verified closure's recovered translation magnitude: no real revisit inside a
# single EuRoC room traverses farther than this. Needed because verification (num_matches,
# reprojection_error) does not by itself catch a degenerate PnP/RANSAC solution -- an
# instrumented sweep on V2_03_difficult found ~1.1% of otherwise-plausible-looking verified pairs
# come back with translation magnitudes in the millions of meters (obvious RANSAC garbage from a
# bad point configuration), and neither of the other two signals flagged them.
LOOP_CLOSURE_MAX_TRANSLATION_M = 20.0

# Noise for an auto-detected closure inserted into the real pose graph (see _run_gtsam's
# enable_loop_closure). Plain Gaussian, not Huber or DCS: both were tested and found to cap or
# outright zero a *correct* closure's pull once the prior disagreement is many multiples of
# sigma, which it always is right after a long blackout -- see the Loop Closure plan's Phase 2/3
# writeup. False-positive protection lives entirely in _LoopClosureDetector's verification gate
# (match count + degeneracy check), not in this noise model.
LOOP_CLOSURE_ROT_SIGMA = 0.02   # rad
LOOP_CLOSURE_TRANS_SIGMA = 0.05  # m


# Candidates this close together in *both* query time and anchor time are treated as the same
# revisit event rather than independent evidence -- chained (each candidate compared to the
# previous one already placed in the cluster, not to the cluster's first member) so a slowly
# drifting match target during a hover/settle still merges into one cluster even though its first
# and last anchors may differ by more than this on their own.
LOOP_CLOSURE_CLUSTER_GAP_S = 2.0


class _LoopClosureDetector:
    """Detects loop closures one keyframe at a time, instead of scanning the whole finalized
    keyframe set in a single batch pass. For every keyframe, checks whether it revisits an
    earlier one: a bag-of-words shortlist (ORBVocabulary; see orb_vocabulary.py) as the cheap
    place-recognition signal, then PnP/RANSAC geometric verification via the same _run_pnp_step
    already used for temporal matching elsewhere in this module. add_keyframe(node, frame) only
    ever searches against nodes 0..node-1, already added -- never one added later.

    The vocabulary itself is a fixed asset trained offline, once, on a *different* sequence than
    whatever this detector runs on (see tmp/investigate/train_orb_vocab.py) -- not something
    derived from this run's own frames, so using it doesn't reintroduce the kind of lookahead the
    rest of this module avoids. What varies per query is only which of the *already-added*
    keyframes' bag-of-words vectors it's compared against.

    Only the single best-scoring earlier candidate is kept per query, and same-event clusters are
    then consolidated into one representative each -- a stay-put stretch has every one of its
    keyframes independently "rediscover" the same historical match, and inserting all of them as
    separate factors overcounts what is really one correlated observation as if it were N
    independent ones (verified: this artificially over-stiffens that part of the graph and
    perturbs other, untouched regions through the shared IMU/bias chain, on a regression check
    against MH_02_easy and MH_04_difficult). A candidate is only returned if it clears
    min_matches (an instrumented precision/recall sweep on V2_03_difficult found ~84-92%
    precision, measured as correct relative-pose recovery rather than raw position proximity, at
    150-300) *and* the translation-magnitude sanity check below.

    Consolidation is a small streaming state machine: a cluster only closes -- and its
    representative (the member with the most temporal matches) becomes a ready _LoopClosureEdge
    -- once a later, non-matching candidate arrives (add_keyframe's return value) or the caller
    calls flush() at the end of the run. That's a delay bounded by LOOP_CLOSURE_CLUSTER_GAP_S
    worth of keyframes, not a whole-sequence lookahead: by the time a cluster closes, its own
    to_node may be several keyframes in the past, but never one not yet added.
    Improved, not resolved -- re-checked against all three regression-check datasets after adding
    this consolidation. MH_04_difficult's regression (from an earlier per-query,
    no-consolidation version) is essentially gone (~12% worse than baseline in its
    known-sensitive ~46s region -> ~1% or better). MH_02_easy is mixed -- some points improved,
    but its worst regression point barely moved and the overall final-trajectory error got worse
    than the unconsolidated version (still better than no closures at all, just not as good). And
    on V2_03_difficult itself (94 closures -> 15), the correction is measurably weaker during the
    transition into the corrected region, though still a >95% reduction from baseline overall. So
    some of what this collapses down was real corrective signal, not pure redundancy --
    collapsing every same-event cluster to exactly one representative is too aggressive in at
    least one direction. LOOP_CLOSURE_CLUSTER_GAP_S=2.0 and "keep only the single best match per
    cluster" are both first guesses, not tuned values; likely next step is keeping a small spread
    of representatives per cluster instead of one.

    The original batch version (see git history) got its speed from a thread pool across all K
    queries at once, which stopped being available once a query only exists after its own keyframe
    arrives -- see zero_lookahead reports in tmp/ for that finding. Bag-of-words replaced that lost
    parallelism with less total work instead: profiled on V2_03_difficult 0-90s, the brute-force
    truncated-descriptor probe this replaced spent 22.6s of 30.6s in knnMatch (44,074 calls, one
    per earlier candidate per query); the inverted-index shortlist here cuts that to ~5 knnMatch
    calls per query (SHORTLIST_N, on the BoW-selected shortlist only), for a ~40% faster
    end-to-end run with equivalent or better accuracy on every regression-check sequence tested
    (MH_02_easy, MH_04_difficult, V2_03_difficult).
    """

    # BoW gives a cheap similarity *ranking*, not a precise match -- same role the old truncated
    # brute-force probe used to play. ORB returns keypoints response-ordered, so the strongest
    # ~PROBE_N carry most of the discriminative signal, same rationale as _scan_keyframes' own
    # PROBE_N=500 probe; truncating here bounds the cost of assigning a keyframe's descriptors to
    # vocabulary words. The BoW ranking alone isn't trusted to pick the actual winner (a coarse
    # signal can rank the truly-best candidate a few slots off) -- so it only builds a shortlist
    # here; SHORTLIST_N candidates are then re-scored with full descriptors (O(SHORTLIST_N) exact
    # comparisons per query, not O(earlier) of them) before picking the winner. Final PnP
    # verification (once per query on that winner) is always full-descriptor and unaffected either
    # way.
    PROBE_N = 500
    SHORTLIST_N = 5

    def __init__(
        self, data: EuRoCMAVData,
        feature_detection_result: FeatureDetectionResult,
        stereo_matching_result: StereoMatchingResult,
        min_temporal_gap_s: float = 10.0,
        min_matches: int = 200,
        vocab: Optional[ORBVocabulary] = None,
    ) -> None:
        self._data = data
        self._fd_result = feature_detection_result
        self._sm_result = stereo_matching_result
        self._min_temporal_gap_s = min_temporal_gap_s
        self._min_matches = min_matches
        self._vocab = vocab if vocab is not None else ORBVocabulary.load()
        self._body_T_cam0 = data.cam0_extrinsics
        self._cam0_T_body = np.linalg.inv(self._body_T_cam0)
        self._first_ts = data.cam_timestamps_ns[0]
        self._kf_times: list[float] = []   # elapsed_s per node added so far
        self._kf_frames: list[int] = []    # original frame index per node added so far
        self._bow_vectors: list[dict[int, float]] = []      # this detector's own BoW vector, per node
        self._inverted_index: dict[int, list[int]] = {}     # vocabulary word id -> nodes containing it
        # (from_node, to_node, candidate) triples of the currently-open, not-yet-closed cluster.
        self._open_cluster: list[tuple[int, int, LoopClosureCandidate]] = []

    def add_keyframe(self, node: int, frame: int) -> list[_LoopClosureEdge]:
        """Search for a revisit at this keyframe against every earlier one already added, then
        feed any match into the consolidation state machine. Returns newly-finalized edges, if a
        cluster just closed as a result -- its own to_node may be several keyframes in the past,
        never one not yet added.
        """
        q_time = _elapsed_s(self._sm_result.frames[frame].timestamp_ns, self._first_ts)
        q_desc_full = self._fd_result.frames[frame].cam0_descriptors
        query_vec = self._vocab.transform(q_desc_full[:self.PROBE_N]) if q_desc_full is not None else {}
        result = self._search(frame, q_time, query_vec)
        # Index this node's own BoW vector *after* searching -- it must never match itself, and
        # only needs to be visible to later queries, never this one.
        self._kf_times.append(q_time)
        self._kf_frames.append(frame)
        self._bow_vectors.append(query_vec)
        for word in query_vec:
            self._inverted_index.setdefault(word, []).append(node)
        if result is None:
            return []
        from_node, candidate = result
        return self._offer(node, from_node, candidate)

    def flush(self) -> list[_LoopClosureEdge]:
        """Call once after the last add_keyframe: closes whatever cluster is still open."""
        return self._close_cluster()

    def _bow_shortlist(self, q_time: float, query_vec: dict[int, float]) -> list[int]:
        """Nodes sharing at least one vocabulary word with the query, ranked by BoW similarity
        (dot product of L2-normalized TF-IDF vectors), via the inverted index -- a candidate with
        zero shared words is never even touched, unlike a brute-force scan of every earlier node.
        """
        scores: dict[int, float] = {}
        for word, qw in query_vec.items():
            for c in self._inverted_index.get(word, ()):
                scores[c] = scores.get(c, 0.0) + qw * self._bow_vectors[c][word]
        eligible = [(s, c) for c, s in scores.items() if q_time - self._kf_times[c] >= self._min_temporal_gap_s]
        eligible.sort(key=lambda x: -x[0])
        return [c for _, c in eligible[:self.SHORTLIST_N]]

    def _search(
        self, frame: int, q_time: float, query_vec: dict[int, float],
    ) -> Optional[tuple[int, LoopClosureCandidate]]:
        if not query_vec:
            return None
        shortlist = self._bow_shortlist(q_time, query_vec)
        if not shortlist:
            return None
        q_desc_full = self._fd_result.frames[frame].cam0_descriptors

        # Re-score the BoW shortlist with full descriptors and pick the true best among them.
        bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        best_score, best_c = -1, -1
        for c in shortlist:
            c_desc_full = self._fd_result.frames[self._kf_frames[c]].cam0_descriptors
            raw = bf.knnMatch(c_desc_full, q_desc_full, k=2)
            score = sum(1 for pair in raw if len(pair) == 2 and pair[0].distance < 0.75 * pair[1].distance)
            if score > best_score:
                best_score, best_c = score, c
        if best_score < self._min_matches:
            return None

        c_frame = self._kf_frames[best_c]
        c_sm = self._sm_result.frames[c_frame]
        c_fd = self._fd_result.frames[c_frame]
        q_fd = self._fd_result.frames[frame]
        if not c_sm.matches:
            return None
        try:
            rvec, tvec, num_matches, reproj_err = _run_pnp_step(
                self._data, c_sm.points_3d, c_sm.matches,
                c_fd.cam0_descriptors, q_fd.cam0_keypoints, q_fd.cam0_descriptors)
        except Exception:
            return None
        if num_matches < self._min_matches or float(np.linalg.norm(tvec)) > LOOP_CLOSURE_MAX_TRANSLATION_M:
            return None

        pnp_cam0 = _rvec_tvec_to_transform(rvec, tvec)
        rel_pose_body = self._body_T_cam0 @ pnp_cam0 @ self._cam0_T_body
        return best_c, LoopClosureCandidate(c_frame, frame, rel_pose_body, num_matches, float(reproj_err))

    def _offer(self, to_node: int, from_node: int, candidate: LoopClosureCandidate) -> list[_LoopClosureEdge]:
        if not self._open_cluster:
            self._open_cluster = [(from_node, to_node, candidate)]
            return []
        prev_from, prev_to, _ = self._open_cluster[-1]
        same_event = (abs(self._kf_times[to_node] - self._kf_times[prev_to]) <= LOOP_CLOSURE_CLUSTER_GAP_S
                      and abs(self._kf_times[from_node] - self._kf_times[prev_from]) <= LOOP_CLOSURE_CLUSTER_GAP_S)
        if same_event:
            self._open_cluster.append((from_node, to_node, candidate))
            return []
        finalized = self._close_cluster()
        self._open_cluster = [(from_node, to_node, candidate)]
        return finalized

    def _close_cluster(self) -> list[_LoopClosureEdge]:
        if not self._open_cluster:
            return []
        best_from, best_to, best = max(self._open_cluster, key=lambda x: x[2].num_matches)
        self._open_cluster = []
        return [_LoopClosureEdge(
            best_from, best_to, best.body_relative_pose,
            LOOP_CLOSURE_ROT_SIGMA, LOOP_CLOSURE_TRANS_SIGMA, "gaussian")]


def _get_pnp_result(
    data: EuRoCMAVData,
    stereo_matching_result: StereoMatchingResult,
    first_timestamp_ns: int,
    min_timestamp_ns: int,
    pnp_poses: list[np.ndarray],
) -> SlamPnpResult:
    # pnp_poses is the per-frame cam0 dead-reckoning trajectory from the shared keyframe scan
    # (_scan_keyframes) -- no separate PnP pass is run here.
    cam_timestamps_ns = np.array([f.timestamp_ns for f in stereo_matching_result.frames])
    # Anchor the trajectory to GT at the first GT sample inside the window (after start_s),
    # not the dataset's first sample -- otherwise the windowed poses get aligned to a GT pose
    # from before the window even begins.
    first_gt_sample = next(s for s in data.ground_truth_samples if s.timestamp_ns >= min_timestamp_ns)
    closest_cam_index = np.argmin(np.abs(cam_timestamps_ns - first_gt_sample.timestamp_ns))

    body_T_cam0 = data.cam0_extrinsics
    cam0_T_body = np.linalg.inv(body_T_cam0)

    world_T_body_first = np.eye(4)
    world_T_body_first[:3, :3] = quaternion_to_rotation_matrix(first_gt_sample.quaternion)
    world_T_body_first[:3, 3] = first_gt_sample.position

    pnp_body_poses = [body_T_cam0 @ T @ cam0_T_body for T in pnp_poses]
    T_comp = world_T_body_first @ np.linalg.inv(pnp_body_poses[closest_cam_index])
    pnp_world_T_body = np.array([T_comp @ T for T in pnp_body_poses])

    pnp_times = np.array([
        _elapsed_s(stereo_matching_result.frames[i].timestamp_ns, first_timestamp_ns)
        for i in range(len(pnp_world_T_body))
    ])

    angular_velocities = []
    for i in range(len(pnp_world_T_body) - 1):
        rotation_matrix = pnp_world_T_body[i, :3, :3].T @ pnp_world_T_body[i + 1, :3, :3]
        rotation_vector, _ = cv2.Rodrigues(rotation_matrix)
        angular_velocities.append(rotation_vector.flatten() * data.cam0_rate_hz)

    rotation_matrices = pnp_world_T_body[:, :3, :3]
    return SlamPnpResult(
        times=pnp_times,
        positions=pnp_world_T_body[:, :3, 3],
        attitudes=_mats_to_rvecs(rotation_matrices),
        rotation_matrices=rotation_matrices,
        angular_velocity_times=pnp_times[:-1],
        angular_velocities=np.array(angular_velocities),
    )


@dataclass(frozen=True)
class LandmarkObservation:
    """A single sighting of one landmark at one keyframe. uv0/uv1 are undistorted pixel
    coordinates in cam0/cam1 (matching a Cal3_S2 pinhole model); point_cam0 is the
    stereo-triangulated 3D point in the cam0 frame at that observation, used only to seed the
    landmark's initial 3D guess -- the shared position estimate itself lives in the factor graph.
    """
    node: int  # graph-node index (into keyframe_indices) that observed the landmark
    uv0: np.ndarray
    uv1: np.ndarray
    point_cam0: np.ndarray


# Match a keyframe's stereo-matched cam0 keypoint to the optical-flow track currently sitting on
# it, if any. A track is seeded from an ORB keypoint (optical_flow.py) and drifts sub-pixel from
# there via KLT, so it rarely lands exactly on a keypoint re-detected fresh at a later frame --
# but a real corner keeps getting re-detected close to where the track still is, so a small pixel
# radius is enough to identify it without any descriptor matching. ORB keypoints can still cluster
# tightly enough that two of them both fall within this radius of the same track (common on richly
# textured regions), so _snap_obs_to_tracks additionally enforces one obs per track. 3.0 was too
# tight on V1_03_difficult's fast-rotation segments (KLT drift between keyframes up to MAX_GAP_S=0.75s
# (15 frames at the nominal 20Hz cam rate) apart exceeded it, dropping real correspondences and starving keyframes of landmarks --
# fewer, sparser tracks hurt rotation accuracy more than the snap collisions they avoided). Kept
# comfortably under optical_flow.MIN_TRACK_SEPARATION_PX (8px) so it still can't straddle two
# distinct live tracks.
LANDMARK_SNAP_PX = 6.0


def _snap_obs_to_tracks(
    fd: FeatureDetectionFrame, sm: StereoMatchingFrame, optical_flow_frame: OpticalFlowFrame,
) -> dict[int, int]:
    """obs index (into sm.matches) -> optical-flow track id, for the nearest live track within
    LANDMARK_SNAP_PX, if any. Each track claims at most one obs (its closest) -- ORB keypoints can
    cluster tightly enough that two of them both land within the snap radius of the same track;
    without this a track could pick up two observations at the same keyframe, which breaks the
    one-observation-per-node assumption _LandmarkTrackBuilder and _run_gtsam rely on.
    """
    if not sm.matches or not optical_flow_frame.track_uv:
        return {}
    obs_pts = np.array([fd.cam0_keypoints[m.queryIdx].pt for m in sm.matches], dtype=np.float32)
    track_ids = list(optical_flow_frame.track_uv.keys())
    track_pts = np.array([optical_flow_frame.track_uv[tid] for tid in track_ids], dtype=np.float32)
    d = np.linalg.norm(obs_pts[:, None, :] - track_pts[None, :, :], axis=2)
    nearest = d.argmin(axis=1)
    nearest_dist = d[np.arange(len(obs_pts)), nearest]
    obs_to_flow_tid: dict[int, int] = {}
    claimed_tracks: set[int] = set()
    for i in np.argsort(nearest_dist):
        if nearest_dist[i] >= LANDMARK_SNAP_PX:
            break
        track_id = track_ids[nearest[i]]
        if track_id in claimed_tracks:
            continue
        obs_to_flow_tid[int(i)] = track_id
        claimed_tracks.add(track_id)
    return obs_to_flow_tid


class _LandmarkTrackBuilder:
    """Chains stereo-matched features into persistent landmark tracks, one keyframe at a time.

    Only cam0-cam1 stereo inliers are tracked, so every observation carries a metric depth for
    initialization; observations are pre-undistorted so they match a Cal3_S2 pinhole model.
    Correspondence across keyframes comes from optical flow's already-tracked ids (a track id is
    valid correspondence through every intermediate frame, not just the two keyframes being
    linked) instead of re-matching ORB descriptors keyframe-to-keyframe -- the same fast-rotation
    failure mode OpticalFlowSolver's docstring documents for that approach.

    Stateful, fed one keyframe at a time via add_keyframe, rather than taking the whole
    keyframe_indices array at once: each call reads only that keyframe's own stereo/feature/
    optical-flow data plus this object's own running state (self.tracks, and which optical-flow
    track id was alive at the *previous* add_keyframe call) -- nothing from a keyframe not yet
    added. There is deliberately no minimum-track-length filtering here: a track's eventual length
    isn't knowable at the node where it starts, so gating on it would reintroduce a lookahead.
    self.tracks holds every chained observation, mature or not; _process_node_landmarks (in
    _run_gtsam) is what decides -- per node, using only that node and earlier -- whether a track is
    mature enough to insert into the graph.
    """

    def __init__(
        self, data: EuRoCMAVData,
        feature_detection_result: FeatureDetectionResult,
        stereo_matching_result: StereoMatchingResult,
        optical_flow_result: OpticalFlowResult,
        depth_min: float, depth_max: float,
    ) -> None:
        self._fd_result = feature_detection_result
        self._sm_result = stereo_matching_result
        self._of_result = optical_flow_result
        self._depth_min = depth_min
        self._depth_max = depth_max
        self._K0 = data.cam0_intrinsics.to_matrix()
        self._K1 = data.cam1_intrinsics.to_matrix()
        self._dist0 = np.array([data.cam0_intrinsics.k1, data.cam0_intrinsics.k2,
                                data.cam0_intrinsics.p1, data.cam0_intrinsics.p2])
        self._dist1 = np.array([data.cam1_intrinsics.k1, data.cam1_intrinsics.k2,
                                data.cam1_intrinsics.p1, data.cam1_intrinsics.p2])
        self.tracks: dict[int, list[LandmarkObservation]] = {}
        self._next_tid = 0
        # optical-flow track id (as of the *last* add_keyframe call) -> our track id.
        self._prev_flow_tid_to_tid: dict[int, int] = {}

    def add_keyframe(self, node: int, frame: int) -> list[int]:
        """Process the keyframe at graph-node index `node` (original frame index `frame`),
        extending self.tracks in place. Returns the track ids that received a new observation at
        this node -- both continuations and freshly started tracks -- so the caller can update its
        own node -> track-id bookkeeping for `node` alone.
        """
        sm = self._sm_result.frames[frame]
        touched: list[int] = []
        if not sm.matches:
            self._prev_flow_tid_to_tid = {}
            return touched
        fd = self._fd_result.frames[frame]
        q_idx = [m.queryIdx for m in sm.matches]
        t_idx = [m.trainIdx for m in sm.matches]
        uv0 = np.array([fd.cam0_keypoints[i].pt for i in q_idx], dtype=np.float64)
        uv1 = np.array([fd.cam1_keypoints[i].pt for i in t_idx], dtype=np.float64)
        uv0u = cv2.undistortPoints(uv0.reshape(-1, 1, 2), self._K0, self._dist0, P=self._K0).reshape(-1, 2)
        uv1u = cv2.undistortPoints(uv1.reshape(-1, 1, 2), self._K1, self._dist1, P=self._K1).reshape(-1, 2)
        pts = sm.points_3d.T  # (M, 3), row i <-> sm.matches[i]

        obs_to_flow_tid = _snap_obs_to_tracks(fd, sm, self._of_result.frames[frame])
        cur_flow_tid_to_tid: dict[int, int] = {}
        for i in range(len(sm.matches)):
            flow_tid = obs_to_flow_tid.get(i)
            if flow_tid is None:
                continue
            landmark_obs = LandmarkObservation(node, uv0u[i], uv1u[i], pts[i])
            tid = self._prev_flow_tid_to_tid.get(flow_tid)
            if tid is None:
                # A brand-new track: gate its depth now, using only this (its first) observation
                # -- the same check the old batch version made from obs[0] once a track's full
                # extent was known, just made at obs[0]'s own time instead of at the end.
                if not (self._depth_min < landmark_obs.point_cam0[2] < self._depth_max):
                    continue
                tid = self._next_tid
                self._next_tid += 1
                self.tracks[tid] = []
            self.tracks[tid].append(landmark_obs)
            cur_flow_tid_to_tid[flow_tid] = tid
            touched.append(tid)
        self._prev_flow_tid_to_tid = cur_flow_tid_to_tid
        return touched


def _scan_keyframes(
    data: EuRoCMAVData,
    feature_detection_result: FeatureDetectionResult,
    stereo_matching_result: StereoMatchingResult,
    N: int,
    on_progress: Optional[Callable[[float], None]] = None,
) -> tuple[list[int], list[np.ndarray]]:
    """Single per-frame PnP scan that yields BOTH the factor-graph nodes and a per-frame PnP
    trajectory, so the ref->frame ORB matching (the SLAM stage's dominant cost) is done once
    rather than separately for keyframe selection and the PnP diagnostic.

    Adaptive keyframing: a single PnP from the current reference keyframe to frame i yields both
    signals we need at once -- the temporal-match count (covisibility with the reference, which
    decays as we move away) and the relative pose (motion since the reference). We open a new
    keyframe when overlap drops below a fraction of the post-keyframe baseline, when enough
    translation/rotation has accrued, or when a hard frame cap is hit -- all clamped by a min gap
    so we never place near-duplicate nodes. Compared to a fixed stride this keeps large,
    well-conditioned baselines when the camera moves fast and avoids zero-parallax nodes when it
    hovers. The chained relative poses are also returned as a per-frame cam0 dead-reckoning
    trajectory for the PnP diagnostic view.
    """
    MIN_GAP = 3                     # never place keyframes closer than this (avoid duplicate nodes)
    # Time-based, not frame-count: a frame-count cap silently doubles its effective time span
    # whenever the camera's actual frame rate drops below the nominal 20Hz it was tuned for (e.g.
    # cam0 dropping to ~10Hz during V2_03_difficult's ~65-75s exposure glitch), which is exactly
    # when this bound on IMU-only preintegration drift matters most. 0.75s = 15 frames at 20Hz,
    # matching the original tuning.
    MAX_GAP_S = 0.75                # force a keyframe at least this often (bound IMU preint. drift)
    COVIS_RATIO = 0.6               # new keyframe once covisibility falls below this * baseline
    TRANS_THRESH = 0.2              # ... or once translation since the reference exceeds this [m]
    ROT_THRESH = np.deg2rad(10.0)   # ... or rotation exceeds this [rad]
    # This probe only needs a covisibility *count* and a rough relative pose to decide where to
    # place nodes -- not a precise reconstruction. ORB brute-force matching is O(features^2) and
    # dominates the whole SLAM stage, so match only the strongest PROBE_N descriptors (ORB returns
    # them response-ordered). The covisibility ratio is scale-free (baseline and current are both
    # capped the same way), so placement is essentially unchanged at a fraction of the cost.
    PROBE_N = 500
    # Never anchor a graph node on a "dead-vision" frame -- one with too few stereo matches to
    # triangulate/observe landmarks (motion blur, low texture). Such a keyframe gets no landmark
    # reprojection factors, so its pose leans entirely on a noisy chained PnP + the IMU factor;
    # ISAM2 then reconciles their disagreement by spiking the gyro bias, corrupting the pose and
    # injecting a permanent trajectory offset (see the ~20 s failure on MH_02_easy). The floor is
    # ~1st percentile of per-frame stereo-match counts, so it only rejects the genuinely starved
    # tail, not ordinary low-overlap keyframes. Decided the moment each candidate is proposed (see
    # _accept_keyframe below), not in a later pass over the finalized list.
    MIN_KF_MATCHES = 80
    # A dead-vision candidate is merged into the ongoing IMU-only gap rather than becoming its own
    # node -- but only up to this bound, so a run of dead-vision candidates can't open an
    # unbounded gap; past it, keep the candidate anyway as the least-bad option, to cap
    # preintegration drift. Time-based (see MAX_GAP_S above) for the same reason.
    DROP_MAX_GAP_S = 2 * MAX_GAP_S

    def _stereo_count(frame_idx: int) -> int:
        return len(stereo_matching_result.frames[frame_idx].matches)

    keyframes = [0]

    def _accept_keyframe(candidate: int) -> None:
        """Add `candidate` as a keyframe, unless it landed on dead vision *and* doing so wouldn't
        open too wide an IMU-only gap -- in which case it's silently merged into the gap instead.
        Uses only `candidate`'s own (already-available) stereo count and the last *kept*
        keyframe's timestamp -- never anything about a candidate not yet proposed. No exemption
        for whichever candidate turns out to be the last one: the same DROP_MAX_GAP_S bound
        that already governs every interior merge governs the tail too (a real-time stream has
        no "last frame" to special-case in the first place).
        """
        elapsed_s = _elapsed_s(stereo_matching_result.frames[candidate].timestamp_ns,
                               stereo_matching_result.frames[keyframes[-1]].timestamp_ns)
        if _stereo_count(candidate) < MIN_KF_MATCHES and elapsed_s <= DROP_MAX_GAP_S:
            return
        keyframes.append(candidate)

    poses: list[Optional[np.ndarray]] = [None] * N
    poses[0] = np.eye(4)
    ref_idx = 0
    ref_covis: Optional[int] = None
    i = 1
    while i < N:
        if on_progress is not None:
            on_progress(i / N)
        ref_fd = feature_detection_result.frames[ref_idx]
        ref_sm = stereo_matching_result.frames[ref_idx]
        cur_fd = feature_detection_result.frames[i]
        try:
            rvec, tvec, num_matches, _ = _run_pnp_step(
                data, ref_sm.points_3d, ref_sm.matches,
                ref_fd.cam0_descriptors[:PROBE_N], cur_fd.cam0_keypoints, cur_fd.cam0_descriptors[:PROBE_N],
            )
        except Exception:
            # Overlap with the reference is gone: anchor a keyframe at the last connected frame
            # (or one past the reference if that is already frame i-1), then re-evaluate i.
            kf = max(i - 1, ref_idx + 1)
            _accept_keyframe(kf)
            if poses[kf] is None:  # this frame never got its own PnP; carry the reference pose
                poses[kf] = poses[ref_idx]
            ref_idx, ref_covis = kf, None
            i = kf + 1
            continue

        step = _rvec_tvec_to_transform(rvec, tvec)
        # poses[ref_idx] is always populated by the time it's read here (poses[0] is seeded
        # above, and ref_idx only ever points at an already-visited frame) -- Pylance can't
        # prove that invariant from the list's `Optional[np.ndarray]` element type.
        poses[i] = poses[ref_idx] @ step  # type: ignore[operator]

        if ref_covis is None:  # first frame after a keyframe sets the covisibility baseline
            ref_covis = num_matches

        gap = i - ref_idx
        translation = float(np.linalg.norm(tvec))
        rotation = float(np.linalg.norm(rvec))
        elapsed_s = _elapsed_s(stereo_matching_result.frames[i].timestamp_ns,
                               stereo_matching_result.frames[ref_idx].timestamp_ns)

        force = elapsed_s >= MAX_GAP_S
        allow = gap >= MIN_GAP
        weak = num_matches < COVIS_RATIO * ref_covis
        moved = translation > TRANS_THRESH or rotation > ROT_THRESH

        if force or (allow and (weak or moved)):
            _accept_keyframe(i)
            ref_idx, ref_covis = i, None
        i += 1

    # No forced keyframe at the last frame of the window, and no exemption from the dead-vision
    # merge either. There used to be both: an unconditional `keyframes.append(N - 1)` if the loop
    # hadn't already placed one there, and a separate pass afterward that dropped dead-vision
    # candidates from everywhere *except* whichever one turned out to be first or last. Both
    # required knowing where the data ends -- true for this offline/windowed tool, never true for
    # a live stream, which just keeps going. Neither is needed: MAX_GAP_S already guarantees a
    # keyframe candidate at least every 0.75s regardless of motion, and DROP_MAX_GAP_S (in
    # _accept_keyframe, applied uniformly now, including to whatever candidate turns out to be
    # last) bounds how much further a dead-vision candidate can push that gap -- so the tail is
    # never more than DROP_MAX_GAP_S short of wherever the window actually ends, a small bounded
    # difference, not an unrepresented trajectory segment. The per-frame dead-reckoning trajectory
    # (`poses`, returned alongside keyframes) still covers every frame up to N-1 regardless; only
    # the GTSAM graph's own last node can land short of the requested window's end. (frame 0 stays
    # unconditionally exempt from the dead-vision check -- not a lookahead issue, since it's
    # simply wherever this scan starts, never a decision made using data from later frames.)

    # Forward-fill any frame that never received a pose (rare: skipped by an exception jump).
    last = np.eye(4)
    filled = []
    for p in poses:
        last = p if p is not None else last
        filled.append(last)
    return keyframes, filled


class _KeyframeScanner:
    """Incremental version of the adaptive-keyframing PnP scan above (_scan_keyframes): the exact
    same algorithm -- a per-frame PnP against a floating reference decides both keyframe placement
    and a per-frame dead-reckoning pose -- restructured so a caller feeds it one frame at a time
    (add_frame) as that frame's fd/sm becomes available, instead of handing it the fully
    materialized [0, N) result up front.

    add_frame(i) requires frame i's own fd/sm to already be present in the
    feature_detection_result/stereo_matching_result passed to __init__ (both are the same
    incrementally-growing containers the caller's own per-frame loop is building) -- a frame after
    i is never touched. Returns the frame index(es), if any, that became a keyframe as a direct
    result of processing frame i: almost always 0 or 1, occasionally 2 -- mirroring
    _scan_keyframes' original while loop, which could resolve an earlier reference frame's fate
    (via the exception-retry path) while handling the frame it was actually called for.
    """
    MIN_GAP = 3
    MAX_GAP_S = 0.75
    COVIS_RATIO = 0.6
    TRANS_THRESH = 0.2
    ROT_THRESH = np.deg2rad(10.0)
    PROBE_N = 500
    MIN_KF_MATCHES = 80
    DROP_MAX_GAP_S = 2 * MAX_GAP_S

    def __init__(
        self, data: EuRoCMAVData,
        feature_detection_result: FeatureDetectionResult,
        stereo_matching_result: StereoMatchingResult,
        n: int,
    ) -> None:
        self._data = data
        self._fd_result = feature_detection_result
        self._sm_result = stereo_matching_result
        self.keyframes: list[int] = [0]
        # Pre-sized (not append-only): the original writes poses[kf] for whatever kf the
        # exception-retry path resolves, which isn't always the highest index reached so far --
        # see add_frame's docstring. Sizing to `n` up front is a memory reservation for this
        # window's known frame count, not a read of any frame's content ahead of its own turn.
        self._poses: list[Optional[np.ndarray]] = [None] * n
        self._poses[0] = np.eye(4)
        self._ref_idx = 0
        self._ref_covis: Optional[int] = None

    def _stereo_count(self, frame_idx: int) -> int:
        return len(self._sm_result.frames[frame_idx].matches)

    def _accept_keyframe(self, candidate: int) -> Optional[int]:
        elapsed_s = _elapsed_s(self._sm_result.frames[candidate].timestamp_ns,
                               self._sm_result.frames[self.keyframes[-1]].timestamp_ns)
        if self._stereo_count(candidate) < self.MIN_KF_MATCHES and elapsed_s <= self.DROP_MAX_GAP_S:
            return None
        self.keyframes.append(candidate)
        return candidate

    def add_frame(self, i: int) -> list[int]:
        accepted: list[int] = []
        while True:
            ref_fd = self._fd_result.frames[self._ref_idx]
            ref_sm = self._sm_result.frames[self._ref_idx]
            cur_fd = self._fd_result.frames[i]
            try:
                rvec, tvec, num_matches, _ = _run_pnp_step(
                    self._data, ref_sm.points_3d, ref_sm.matches,
                    ref_fd.cam0_descriptors[:self.PROBE_N], cur_fd.cam0_keypoints,
                    cur_fd.cam0_descriptors[:self.PROBE_N],
                )
            except Exception:
                # Overlap with the reference is gone: anchor a keyframe at the last connected
                # frame (or one past the reference if that is already frame i-1). kf is always
                # exactly i-1 or i (never less -- see _scan_keyframes' identical branch), so
                # either we still owe frame i its own resolution (kf == i-1: retry against the
                # new reference) or frame i's fate was just decided as a side effect (kf == i:
                # nothing more to do for this call).
                kf = max(i - 1, self._ref_idx + 1)
                got = self._accept_keyframe(kf)
                if got is not None:
                    accepted.append(got)
                if self._poses[kf] is None:  # never got its own PnP; carry the reference pose
                    self._poses[kf] = self._poses[self._ref_idx]
                self._ref_idx, self._ref_covis = kf, None
                if kf >= i:
                    return accepted
                continue

            step = _rvec_tvec_to_transform(rvec, tvec)
            # self._poses[self._ref_idx] is always populated by the time it's read here (index 0
            # is seeded in __init__, and _ref_idx only ever points at an already-visited frame).
            self._poses[i] = self._poses[self._ref_idx] @ step  # type: ignore[operator]

            if self._ref_covis is None:  # first frame after a keyframe sets the covisibility baseline
                self._ref_covis = num_matches

            gap = i - self._ref_idx
            translation = float(np.linalg.norm(tvec))
            rotation = float(np.linalg.norm(rvec))
            elapsed_s = _elapsed_s(self._sm_result.frames[i].timestamp_ns,
                                   self._sm_result.frames[self._ref_idx].timestamp_ns)

            force = elapsed_s >= self.MAX_GAP_S
            allow = gap >= self.MIN_GAP
            weak = num_matches < self.COVIS_RATIO * self._ref_covis
            moved = translation > self.TRANS_THRESH or rotation > self.ROT_THRESH

            if force or (allow and (weak or moved)):
                got = self._accept_keyframe(i)
                if got is not None:
                    accepted.append(got)
                self._ref_idx, self._ref_covis = i, None
            return accepted

    @staticmethod
    def _forward_fill(poses: list[Optional[np.ndarray]]) -> list[np.ndarray]:
        last = np.eye(4)
        filled = []
        for p in poses:
            last = p if p is not None else last
            filled.append(last)
        return filled

    def current_poses(self, up_to_frame: int) -> list[np.ndarray]:
        """Forward-filled poses for frames [0, up_to_frame] -- unlike finalize(), safe to call
        mid-scan (before every frame in the window has been fed via add_frame), for a caller that
        wants to render the dead-reckoning trajectory as it grows.
        """
        return self._forward_fill(self._poses[:up_to_frame + 1])

    def finalize(self) -> list[np.ndarray]:
        """Call once after every frame in the window has been fed via add_frame: forward-fills
        any frame that never received its own pose (rare -- see add_frame's exception path)."""
        return self._forward_fill(self._poses)


def _run_gtsam(
    data: EuRoCMAVData,
    feature_detection_result: FeatureDetectionResult,
    stereo_matching_result: StereoMatchingResult,
    optical_flow_result: OpticalFlowResult,
    imu_samples: list[ImuSample],
    gravity: np.ndarray,
    keyframe_indices: list[int],
    on_progress: Callable[[float, str], None],
    # Runs automated loop-closure detection (_LoopClosureDetector) and inserts whatever it finds
    # into this graph, using LOOP_CLOSURE_ROT_SIGMA/TRANS_SIGMA with plain Gaussian noise. Off by
    # default -- see SlamSolver.__init__'s enable_loop_closure for why. Ignored if
    # extra_loop_closures below is given explicitly (manual override always wins over
    # auto-detection).
    enable_loop_closure: bool = False,
    # Manual override / test hook (see tmp/investigate/v2_03_loop_closure_phase*.py). Frame
    # indices (not node indices) -- resolved to nodes internally once keyframe_indices is final.
    # noise_mode in {"gaussian", "huber", "dcs"} -- use "gaussian"; see the noise construction
    # below (at the point of use) for why the other two underperform.
    # Defaults to a no-op: omitting both this and enable_loop_closure leaves _run_gtsam's output
    # byte-for-byte unchanged.
    extra_loop_closures: Optional[list[LoopClosureFactor]] = None,
    # Test-only introspection hook (see v2_03_loop_closure_phase2b.py): if provided, filled with
    # {'isam2': isam2} so a caller can run its own batch re-optimization over the full factor
    # graph afterward, without _run_gtsam's return signature changing for existing callers.
    debug_out: Optional[dict] = None,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], np.ndarray, np.ndarray, list[int]]:
    N = len(stereo_matching_result.frames)
    imu_timestamps_ns = np.array([s.timestamp_ns for s in imu_samples])
    # Mark read-only: these absolute-ns timestamps are a lookup table we only ever *read* (via
    # `imu_timestamps_ns - cam_timestamps_ns[0]` etc.). Without this, NumPy's temporary-elision
    # optimization can execute that subtraction in place -- when this array is large (long IMU
    # prefix) and has refcount 1, `a - scalar` mutates `a` and returns it -- silently turning
    # every timestamp relative, emptying every IMU window, and producing zero-dt (degenerate)
    # ImuFactors that make ISAM2 indeterminate. Read-only arrays are never elided.
    imu_timestamps_ns.flags.writeable = False
    imu_lin_accs      = np.array([s.linear_acceleration for s in imu_samples])
    imu_ang_vels      = np.array([s.angular_velocity for s in imu_samples])
    cam_timestamps_ns = np.array([f.timestamp_ns for f in stereo_matching_result.frames[:N]])

    body_T_cam0 = data.cam0_extrinsics
    cam0_T_body = np.linalg.inv(body_T_cam0)

    # Estimate the gravity direction in the body frame *at frame 0* by averaging the
    # accelerometer over a short window there (at rest it senses specific force, "up").
    # A global most-static window would measure gravity at a different body orientation,
    # because the body is generally rotated differently by the time it settles; integrating
    # the gyro across that gap drifts too much to correct for it reliably.
    grav_win = max(1, int(data.imu0_rate_hz * 0.5))
    i0 = int(np.argmin(np.abs(imu_timestamps_ns - cam_timestamps_ns[0])))
    gravity_in_body = imu_lin_accs[i0:i0 + grav_win].mean(axis=0)

    # Progress budget across this stage's sub-steps (fractions of the GTSAM phase):
    #   setup                [0.00, 0.10]
    #   ISAM2 forward loop   [0.10, 0.92]
    #   reprojection metrics [0.92, 1.00]
    # No separate loop-closure-search slice: detection (when enabled) runs interleaved with the
    # ISAM2 loop, one query per new keyframe, rather than as an upfront batch pass -- so its cost
    # is already spread across the same range the loop's own progress reports advance through.
    # keyframe_indices (from the shared _scan_keyframes pass) maps graph node j -> original frame
    # index; gaps between them vary since nodes are chosen adaptively (covisibility + motion).
    K = len(keyframe_indices)

    X = lambda i: gtsam.symbol('x', i)
    V = lambda i: gtsam.symbol('v', i)
    B = lambda i: gtsam.symbol('b', i)
    L = lambda i: gtsam.symbol('l', i)

    imu_params = gtsam.PreintegrationParams(gravity)
    imu_params.setGyroscopeCovariance(np.eye(3) * 1e-4)
    # Keep the accelerometer factor from over-dominating the PnP between-factors; too small
    # a covariance lets the IMU dead-reckon (and drift) despite accurate vision constraints.
    imu_params.setAccelerometerCovariance(np.eye(3) * 1e-3)
    imu_params.setIntegrationCovariance(np.eye(3) * 1e-8)

    PRIOR_POSE_NOISE = gtsam.noiseModel.Isotropic.Sigma(6, 0.1)
    PRIOR_VEL_NOISE  = gtsam.noiseModel.Isotropic.Sigma(3, 0.1)
    # Loose enough that the optimizer can actually estimate the (constant) IMU bias. A tight
    # prior here pins bias to zero, so the real accelerometer bias double-integrates into drift.
    PRIOR_BIAS_NOISE = gtsam.noiseModel.Isotropic.Sigma(6, 0.1)
    # Per-step (adjacent-frame) PnP sigmas. A keyframe constraint chains the intermediate steps,
    # so its error accumulates like a random walk -> scale the sigmas by sqrt(gap) per interval,
    # where gap is that interval's frame count. Built inside the loop since gaps now vary.
    PNP_STEP_SIGMAS  = np.array([0.01, 0.01, 0.01, 0.05, 0.05, 0.05])
    # Random-walk noise letting the (per-keyframe) IMU bias evolve between keyframes instead of
    # being pinned to a single constant. The accumulated std over an interval of duration dt is
    # (diffusion_density * sqrt(dt)) per axis, so it is built inside the loop (keyframe gaps vary).
    # Both terms use the IMU datasheet's physical diffusion (EuRoC ADIS16448, imu0/sensor.yaml).
    # The gyro term was tightened first -- the old blanket sigma=0.1 permitted ~1 rad/s jumps per
    # keyframe and let the optimizer absorb a pose/rotation error by spiking the gyro bias (the
    # ~20 s MH_02_easy failure), forcing that residual into the (vision-correctable) pose instead.
    # The accel term used to stay at the same loose blanket 0.1 (with no sqrt(dt) scaling at all)
    # to "soak up unmodeled gravity/scale error" -- but that let accel bias free-walk in axes vision
    # can't correct (e.g. z on a near-level flight segment, where there's almost no vertical
    # parallax to check it against), producing a near-linear position drift over tens of seconds.
    # Tightened to the same physical-diffusion treatment as gyro.
    GYRO_BIAS_RW  = 1.9393e-05   # [rad/s^2/sqrt(Hz)]  gyroscope_random_walk from imu0/sensor.yaml
    ACCEL_BIAS_RW = 3.0000e-03   # [m/s^3/sqrt(Hz)]    accelerometer_random_walk from imu0/sensor.yaml
    # Fallback regularizer for a keyframe that ends up with neither a PnP between-factor nor any
    # landmark reprojection factor -- e.g. fast motion where feature tracks break and the chained
    # PnP fails. Such a pose would hang off only its IMU factor (yaw about gravity unobservable),
    # making ISAM2's system indeterminate. A weak prior at the IMU-predicted pose pins the free
    # directions without fighting real constraints; it is far looser than PnP/reprojection noise.
    FALLBACK_POSE_NOISE = gtsam.noiseModel.Isotropic.Sigma(6, 1.0)

    # --- Landmarks (rung 2: explicit structure / bundle adjustment) ---------------------
    # Persistent 3D landmarks tied to poses by reprojection factors. A point seen across
    # several keyframes becomes one shared variable with many rigid constraints (errors
    # average), instead of a chain of noisy relative PnP poses (errors compound).
    MIN_TRACK_LEN   = 3       # observations before a landmark is trusted enough to add
    PNP_FALLBACK_COVIS = 15   # if a keyframe pair shares >= this many landmarks, drop PnP
    PX_SIGMA        = 1.5     # reprojection sigma [px]
    DEPTH_MIN, DEPTH_MAX = 0.3, 40.0
    # cam0/cam1 pinhole calibrations (measurements are pre-undistorted) and body<-cam poses.
    cam0_K = gtsam.Cal3_S2(data.cam0_intrinsics.fx, data.cam0_intrinsics.fy, 0.0,
                           data.cam0_intrinsics.cx, data.cam0_intrinsics.cy)
    cam1_K = gtsam.Cal3_S2(data.cam1_intrinsics.fx, data.cam1_intrinsics.fy, 0.0,
                           data.cam1_intrinsics.cx, data.cam1_intrinsics.cy)
    cam0_pose = gtsam.Pose3(body_T_cam0)
    cam1_pose = gtsam.Pose3(data.cam1_extrinsics)
    # Robust (Huber) pixel noise so a single bad match can't drag a landmark or a pose.
    PX_NOISE = gtsam.noiseModel.Robust.Create(
        gtsam.noiseModel.mEstimator.Huber.Create(1.345),
        gtsam.noiseModel.Isotropic.Sigma(2, PX_SIGMA))
    # Weak prior anchoring each landmark to its stereo-triangulated init. Far / low-parallax
    # points are barely constrained along the viewing ray and would make ISAM2's system
    # indeterminate; this regularizes them. It is orders of magnitude looser than the pixel
    # factors, so it is negligible for well-observed landmarks.
    LM_PRIOR_NOISE = gtsam.noiseModel.Isotropic.Sigma(3, 5.0)

    on_progress(0.0, "Building landmark tracks...")
    # Fed one keyframe at a time below (node 0 here, node j+1 inside _add_keyframe_to_graph, right
    # before that node's factors are built) instead of handed the whole keyframe_indices array up
    # front -- see _LandmarkTrackBuilder's docstring. `tracks` aliases the builder's own dict, so
    # later add_keyframe calls are visible through it without any extra bookkeeping.
    track_builder = _LandmarkTrackBuilder(
        data, feature_detection_result, stereo_matching_result, optical_flow_result,
        DEPTH_MIN, DEPTH_MAX)
    tracks = track_builder.tracks

    # No landmark-starvation keyframe drop here. There used to be one: a keyframe whose per-node
    # landmark-track count fell under a MIN_KF_LANDMARKS threshold got merged into the IMU-only
    # gap between its neighbours, on the theory that a node with too few reprojection factors
    # hangs entirely off the IMU factor and lets ISAM2 explain the shortfall by spiking the bias
    # instead of the pose (see the accel-bias runaway this caused on MH_04_difficult ~46s,
    # historically). Two problems with it:
    #   - It's non-causal even in its best form: whether a track clears MIN_TRACK_LEN isn't known
    #     until MIN_TRACK_LEN-1 keyframes after it starts (tracks are contiguous runs, so this is
    #     bounded, not unbounded -- but it's still a lookahead a real-time caller would have to
    #     buffer for).
    #   - It's no longer needed for the failure mode it was built for. _add_keyframe_to_graph
    #     already has FALLBACK_POSE_NOISE (below) as a weak prior for exactly the case this gate
    #     was trying to avoid -- a node with neither a PnP between-factor nor any reprojection
    #     factor. Tested removing this gate entirely (keep every _scan_keyframes keyframe, causal
    #     by construction since there's no drop decision to make) against the two sequences that
    #     originally motivated it plus two more: mean ATE improved or was statistically flat on
    #     all four (MH_02_easy 0.042m -> 0.019m, MH_04_difficult 0.333m -> 0.224m,
    #     MH_05_difficult 0.521m -> 0.532m, V2_03_difficult 3.435m -> 0.514m). No regression found.
    #   Trade-off: keeps more keyframes (e.g. 391 vs 207 on V2_03_difficult's 0-200s window), so
    #   more ISAM2 work per run.

    # Manual extra_loop_closures (if given) always overrides auto-detection rather than adding to
    # it -- keeps the test/override path from Phase 2/3 exact and predictable. It's a batch,
    # fully-offline API by nature (the caller already knows every closure up front), so it's
    # resolved to node indices here, all at once, same as it always was.
    did_auto_detect_loop_closures = enable_loop_closure and extra_loop_closures is None
    loop_closures_by_node: dict[int, list[_LoopClosureEdge]] = {}
    if extra_loop_closures:
        kf_arr = np.array(keyframe_indices)
        for factor in extra_loop_closures:
            from_node = int(np.argmin(np.abs(kf_arr - factor.from_frame)))
            to_node = int(np.argmin(np.abs(kf_arr - factor.to_frame)))
            loop_closures_by_node.setdefault(to_node, []).append(_LoopClosureEdge(
                from_node, to_node, factor.body_relative_pose, factor.rot_sigma, factor.trans_sigma,
                factor.noise_mode))

    # Auto-detection, when it runs, is fed one keyframe at a time from inside the ISAM2 loop below
    # (loop_detector.add_keyframe, alongside track_builder.add_keyframe) instead of scanning the
    # whole finalized keyframe set up front -- see _LoopClosureDetector's docstring.
    loop_detector = (
        _LoopClosureDetector(data, feature_detection_result, stereo_matching_result)
        if did_auto_detect_loop_closures else None)

    def _add_loop_closure_edges(factors: gtsam.NonlinearFactorGraph, edges: list[_LoopClosureEdge]) -> None:
        for edge in edges:
            delta = _to_gtsam_pose3(edge.body_relative_pose)
            base_noise = gtsam.noiseModel.Diagonal.Sigmas(
                np.array([edge.rot_sigma] * 3 + [edge.trans_sigma] * 3))
            if edge.noise_mode == "huber":
                # Linear (not quadratic) loss beyond ~1.35 sigma -- caps a *correct* closure's
                # pull once the prior disagreement is many multiples of sigma, which it always is
                # right after a long blackout (measured: ~30% of the correction "gaussian" gets).
                # Left in as an option; "gaussian" is what's actually used by auto-detection.
                loop_noise = gtsam.noiseModel.Robust.Create(
                    gtsam.noiseModel.mEstimator.Huber.Create(1.345), base_noise)
            elif edge.noise_mode == "dcs":
                # DCS's scale factor is min(1, 2c / (c + ||r||^2)) where ||r||^2 is the whitened
                # (chi-squared) residual -- so c must be calibrated to this factor's degrees of
                # freedom (6, for a Pose3 BetweenFactor), not left at the textbook-example c=1.
                # A genuinely-consistent 6-DOF residual has ||r||^2 around 6 on average; c=1
                # measured this down to a ~0.29 scale for an *already-correct* closure (verified
                # empirically: it made a 94-closure test on V2_03_difficult behave identically to
                # having no closures at all). c=6 keeps full trust up to the expected value and
                # only meaningfully down-weights past the ~12.6 (95th-percentile chi2_6) mark --
                # i.e. genuine outliers, not closures merely large-but-correct after a long
                # blackout.
                loop_noise = gtsam.noiseModel.Robust.Create(
                    gtsam.noiseModel.mEstimator.DCS.Create(6.0), base_noise)  # type: ignore[attr-defined]
            else:
                loop_noise = base_noise
            factors.add(gtsam.BetweenFactorPose3(X(edge.from_node), X(edge.to_node), delta, loop_noise))

    # node -> track ids observed there; and per-interval covisibility for the PnP gate. Filled
    # incrementally as each node is added below (node 0 here, node j+1 inside
    # _add_keyframe_to_graph) rather than derived from the full `tracks` dict up front.
    nodes_to_tracks: dict[int, list[int]] = {jj: [] for jj in range(K)}
    node_seen: list[set[int]] = [set() for _ in range(K)]

    def _register_node_tracks(node: int, touched: list[int]) -> None:
        nodes_to_tracks[node].extend(touched)
        node_seen[node].update(touched)

    _register_node_tracks(0, track_builder.add_keyframe(0, keyframe_indices[0]))
    if loop_detector is not None:
        loop_detector.add_keyframe(0, keyframe_indices[0])  # never closes a cluster on its own

    inserted_landmarks: set[int] = set()
    added_obs: set[tuple[int, int]] = set()
    n_proj_factors = 0

    def _add_obs_factors(factors: gtsam.NonlinearFactorGraph, tid: int, node: int,
                         uv0: np.ndarray, uv1: np.ndarray) -> None:
        nonlocal n_proj_factors
        factors.add(gtsam.GenericProjectionFactorCal3_S2(
            uv0, PX_NOISE, X(node), L(tid), cam0_K, False, False, cam0_pose))
        factors.add(gtsam.GenericProjectionFactorCal3_S2(
            uv1, PX_NOISE, X(node), L(tid), cam1_K, False, False, cam1_pose))
        added_obs.add((tid, node))
        n_proj_factors += 2

    def _process_node_landmarks(jj: int, est: gtsam.Values,
                                factors: gtsam.NonlinearFactorGraph,
                                values: gtsam.Values) -> int:
        """Add reprojection factors for landmarks observed at node jj (X(jj) already staged).

        Returns how many landmark observations were staged *at node jj* -- i.e. how many
        projection factors now reference X(jj). The caller uses this to tell whether the new
        pose picked up any structure constraint (see the underconstrained-pose guard below).
        """
        n_at_node = 0
        for tid in nodes_to_tracks[jj]:
            obs = tracks[tid]
            if tid in inserted_landmarks:
                if (tid, jj) not in added_obs:
                    o = next(o for o in obs if o.node == jj)
                    _add_obs_factors(factors, tid, jj, o.uv0, o.uv1)
                    n_at_node += 1
                continue
            avail = [o for o in obs if o.node <= jj]
            if len(avail) < MIN_TRACK_LEN:
                continue
            # Initialize the landmark in the nav frame from the first observation's stereo depth.
            first = obs[0]
            T_G_cam0 = est.atPose3(X(first.node)).matrix() @ body_T_cam0
            p_world = (T_G_cam0 @ np.append(first.point_cam0, 1.0))[:3]
            values.insert(L(tid), gtsam.Point3(*p_world))
            factors.add(gtsam.PriorFactorPoint3(L(tid), gtsam.Point3(*p_world), LM_PRIOR_NOISE))
            inserted_landmarks.add(tid)
            for o in avail:
                _add_obs_factors(factors, tid, o.node, o.uv0, o.uv1)
                if o.node == jj:
                    n_at_node += 1
        return n_at_node

    isam2 = gtsam.ISAM2(gtsam.ISAM2Params())

    # Anchor the GTSAM navigation frame G to be gravity-aligned: rotate body-frame-0
    # so its measured gravity lands on the nav-frame gravity axis. At rest the
    # accelerometer reads "up" (specific force), so R_G_body0 @ gravity_in_body = -gravity.
    R_G_body0 = _align_vectors(gravity_in_body, -gravity)
    pose0 = gtsam.Pose3(gtsam.Rot3(R_G_body0), gtsam.Point3(0.0, 0.0, 0.0))

    f0, v0 = gtsam.NonlinearFactorGraph(), gtsam.Values()
    f0.add(gtsam.PriorFactorPose3(X(0), pose0, PRIOR_POSE_NOISE))
    f0.add(gtsam.PriorFactorVector(V(0), np.zeros(3), PRIOR_VEL_NOISE))
    f0.add(gtsam.PriorFactorConstantBias(B(0), gtsam.imuBias.ConstantBias(), PRIOR_BIAS_NOISE))
    v0.insert(X(0), pose0)
    v0.insert(V(0), np.zeros(3))
    v0.insert(B(0), gtsam.imuBias.ConstantBias())
    isam2.update(f0, v0)

    def _add_keyframe_to_graph(j: int) -> None:
        """Add keyframe j+1 to the graph: an IMU factor from the preintegrated window since
        keyframe j, a chained-PnP between-factor when landmark covisibility isn't yet strong
        enough to constrain the pose on its own, reprojection factors for any landmarks observed
        at the new keyframe, and any loop-closure factors targeting it -- then commit the update.
        """
        est = isam2.calculateEstimate()
        pose_i = est.atPose3(X(j))
        vel_i  = est.atVector(V(j))
        bias_i = est.atConstantBias(B(j))

        kf_i, kf_next = keyframe_indices[j], keyframe_indices[j + 1]

        # Extend landmark tracks to this node before anything below reads node_seen[j+1] or
        # tracks[tid] for it (strong_covis just below, then _process_node_landmarks).
        _register_node_tracks(j + 1, track_builder.add_keyframe(j + 1, keyframe_indices[j + 1]))
        # Same for auto-detected loop closures: search for a revisit at this keyframe now, against
        # whatever's been added so far. Anything this returns is edge(s) whose consolidation
        # cluster just closed -- their own to_node may be an earlier node than j+1 (added and
        # optimized in a previous update), not necessarily this one.
        new_loop_edges = loop_detector.add_keyframe(j + 1, keyframe_indices[j + 1]) if loop_detector is not None else []

        new_factors, new_values = gtsam.NonlinearFactorGraph(), gtsam.Values()

        pim = gtsam.PreintegratedImuMeasurements(imu_params, bias_i)
        window = np.where(
            (imu_timestamps_ns >= cam_timestamps_ns[kf_i]) &
            (imu_timestamps_ns <  cam_timestamps_ns[kf_next])
        )[0]
        for k in window:
            dt = (float(imu_timestamps_ns[k + 1] - imu_timestamps_ns[k]) * 1e-9
                  if k + 1 < len(imu_timestamps_ns) else 1.0 / data.imu0_rate_hz)
            pim.integrateMeasurement(imu_lin_accs[k], imu_ang_vels[k], dt)

        new_factors.add(gtsam.ImuFactor(X(j), V(j), X(j + 1), V(j + 1), B(j), pim))
        # Bias random-walk over this interval's duration: sigma = density * sqrt(dt) per axis,
        # ordered [accel(3), gyro(3)] to match imuBias.ConstantBias's tangent.
        dt_kf = float(cam_timestamps_ns[kf_next] - cam_timestamps_ns[kf_i]) * 1e-9
        bias_between_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array(
            [ACCEL_BIAS_RW * np.sqrt(dt_kf)] * 3 + [GYRO_BIAS_RW * np.sqrt(dt_kf)] * 3))
        new_factors.add(gtsam.BetweenFactorConstantBias(
            B(j), B(j + 1), gtsam.imuBias.ConstantBias(), bias_between_noise))

        nav_j     = pim.predict(gtsam.NavState(pose_i, vel_i), bias_i)
        pose_init = nav_j.pose()
        vel_init  = nav_j.velocity()

        # Build the keyframe->keyframe PnP constraint by chaining the intermediate
        # adjacent-frame PnP steps. Each step is between neighbouring frames, where feature
        # overlap is high, so it is well-conditioned; a single direct match across the full
        # ~10-frame gap would have little overlap and fail often. If any step fails, drop the
        # whole constraint for this interval and let the IMU factor carry it.
        #
        # PnP is now a *fallback*: when this keyframe pair already shares plenty of *inserted*
        # landmarks, their reprojection factors constrain the relative pose directly, so adding
        # PnP on top would double-count the same pixels. Only count landmarks that are already
        # mature (inserted) and seen at both endpoints -- those are the ones that actually
        # contribute factors linking X(j) and X(j+1) after this update. A track that is merely
        # covisible but not yet mature contributes nothing here, so PnP must still carry it,
        # otherwise the new pose would be left underconstrained (indeterminate system).
        strong_covis = sum(
            1 for tid in (node_seen[j] & node_seen[j + 1]) if tid in inserted_landmarks)
        pnp_added = False
        if strong_covis < PNP_FALLBACK_COVIS:
            pnp_cam0 = np.eye(4)
            pnp_ok = True
            for f in range(kf_i, kf_next):
                sm_f = stereo_matching_result.frames[f]
                fd_f, fd_f1 = feature_detection_result.frames[f], feature_detection_result.frames[f + 1]
                try:
                    rvec, tvec, _, _ = _run_pnp_step(
                        data, sm_f.points_3d, sm_f.matches,
                        fd_f.cam0_descriptors, fd_f1.cam0_keypoints, fd_f1.cam0_descriptors,
                    )
                except Exception:
                    pnp_ok = False
                    break
                pnp_cam0 = pnp_cam0 @ _rvec_tvec_to_transform(rvec, tvec)

            if pnp_ok:
                pnp_body = body_T_cam0 @ pnp_cam0 @ cam0_T_body
                pnp_delta = _to_gtsam_pose3(pnp_body)
                pnp_noise = gtsam.noiseModel.Diagonal.Sigmas(PNP_STEP_SIGMAS * np.sqrt(kf_next - kf_i))
                new_factors.add(gtsam.BetweenFactorPose3(X(j), X(j + 1), pnp_delta, pnp_noise))
                pose_init = pose_i.compose(pnp_delta)
                pnp_added = True

        new_values.insert(X(j + 1), pose_init)
        new_values.insert(V(j + 1), vel_init)
        new_values.insert(B(j + 1), bias_i)
        # Reprojection factors for landmarks observed at the new keyframe. X(j+1) is staged in
        # new_values (valid within this same update); landmark inits use poses from `est`, all
        # of which predate node j+1, so they are already in the estimate.
        n_proj_at_next = _process_node_landmarks(j + 1, est, new_factors, new_values)
        # Guard: never add a keyframe without a relative constraint. If neither the PnP fallback
        # nor any landmark reprojection factor touched X(j+1), it would hang off only its IMU
        # factor and make ISAM2 indeterminate. Anchor it with a weak prior at the IMU prediction.
        if not pnp_added and n_proj_at_next == 0:
            new_factors.add(gtsam.PriorFactorPose3(X(j + 1), pose_init, FALLBACK_POSE_NOISE))

        # Pre-resolved manual edges targeting exactly this node, plus whatever auto-detected edges
        # just became ready above (which may target this node or an earlier one).
        _add_loop_closure_edges(new_factors, loop_closures_by_node.pop(j + 1, []))
        _add_loop_closure_edges(new_factors, new_loop_edges)

        isam2.update(new_factors, new_values)

    for j in range(K - 1):
        on_progress(0.10 + (j / (K - 1)) * (0.92 - 0.10), "Optimizing (ISAM2)...")
        _add_keyframe_to_graph(j)

    if loop_detector is not None:
        # Whatever cluster was still open when the run ended (its to_node is necessarily <= K-1,
        # already in the graph) -- add_keyframe only closes a cluster when a *later* candidate
        # arrives, so the very last one never gets that chance without this.
        final_edges = loop_detector.flush()
        if final_edges:
            tail_factors = gtsam.NonlinearFactorGraph()
            _add_loop_closure_edges(tail_factors, final_edges)
            isam2.update(tail_factors, gtsam.Values())

    print(f"landmarks: {len(inserted_landmarks)}/{len(tracks)} tracks used, "
          f"{n_proj_factors} reprojection factors")
    on_progress(0.92, "Computing metrics...")
    final = isam2.calculateEstimate()
    poses = [final.atPose3(X(j)).matrix() for j in range(K)]
    velocities = [final.atVector(V(j)) for j in range(K)]
    biases = [final.atConstantBias(B(j)).vector() for j in range(K)]  # [accel(3), gyro(3)]

    # Per-keyframe landmark-quality metrics: reprojection RMSE [px] and how many landmarks were
    # observed there. Reproject each final landmark into every keyframe that saw it, through
    # both cameras, and compare to the (undistorted) measurement.
    inv_Twc0 = [np.linalg.inv(pm @ body_T_cam0) for pm in poses]
    inv_Twc1 = [np.linalg.inv(pm @ data.cam1_extrinsics) for pm in poses]
    sq_px = np.zeros(K)
    n_px = np.zeros(K)
    n_lm = np.zeros(K)
    for tid in inserted_landmarks:
        p = np.append(np.asarray(final.atPoint3(L(tid))), 1.0)
        for o in tracks[tid]:
            seen = False
            for intrin, inv_Twc, uv in (
                (data.cam0_intrinsics, inv_Twc0, o.uv0),
                (data.cam1_intrinsics, inv_Twc1, o.uv1),
            ):
                pc = inv_Twc[o.node] @ p
                if pc[2] <= 1e-6:
                    continue
                u = intrin.fx * pc[0] / pc[2] + intrin.cx
                v = intrin.fy * pc[1] / pc[2] + intrin.cy
                sq_px[o.node] += (u - uv[0]) ** 2 + (v - uv[1]) ** 2
                n_px[o.node] += 1
                seen = True
            if seen:
                n_lm[o.node] += 1
    reprojection_rmse = np.where(n_px > 0, np.sqrt(sq_px / np.maximum(n_px, 1)), np.nan)

    if debug_out is not None:
        debug_out['isam2'] = isam2

    return poses, velocities, biases, reprojection_rmse, n_lm, keyframe_indices


# How many trailing nodes _GtsamBuilder.current_estimate(recompute_window=...) refreshes on each
# live-preview call (see that method's docstring). ~2x the ~15-20 keyframes empirically observed
# for a settled node's ISAM2-revised pose/reprojection error to converge, as safety margin.
_LIVE_PREVIEW_REPROJECTION_WINDOW = 40


class _GtsamBuilder:
    """Incremental version of the ISAM2 graph construction in _run_gtsam above: the exact same
    per-keyframe update (IMU factor, chained-PnP fallback, landmark reprojection factors,
    loop-closure factors), restructured so a caller feeds it one *new* keyframe at a time
    (add_keyframe) as soon as _KeyframeScanner confirms it, instead of handing it a complete
    keyframe_indices list up front.

    add_keyframe(frame_idx) requires every frame's fd/sm from the previous keyframe through
    frame_idx to already be present in the feature_detection_result/stereo_matching_result passed
    to __init__ (needed for the chained-PnP fallback, which steps through every intermediate
    frame) -- a frame after frame_idx is never touched. optical_flow_result only needs frame_idx's
    own entry (track_builder/loop_detector index by frame, not by interval).

    Scope note: unlike _run_gtsam, this does not support the manual extra_loop_closures/debug_out
    test hooks -- those are an inherently batch, fully-offline API (the caller already knows every
    closure up front; see LoopClosureFactor's docstring) used only by tmp/investigate's phase2/
    phase3 scripts, which call _run_gtsam directly and are unaffected by this class existing
    alongside it.
    """

    def __init__(
        self, data: EuRoCMAVData,
        feature_detection_result: FeatureDetectionResult,
        stereo_matching_result: StereoMatchingResult,
        optical_flow_result: OpticalFlowResult,
        imu_samples: list[ImuSample],
        gravity: np.ndarray,
        first_keyframe_frame: int,
        enable_loop_closure: bool = False,
    ) -> None:
        self._data = data
        self._fd_result = feature_detection_result
        self._sm_result = stereo_matching_result
        self._of_result = optical_flow_result

        imu_timestamps_ns = np.array([s.timestamp_ns for s in imu_samples])
        # See _run_gtsam's identical guard for why this must be read-only.
        imu_timestamps_ns.flags.writeable = False
        self._imu_timestamps_ns = imu_timestamps_ns
        self._imu_lin_accs = np.array([s.linear_acceleration for s in imu_samples])
        self._imu_ang_vels = np.array([s.angular_velocity for s in imu_samples])

        self._body_T_cam0 = data.cam0_extrinsics
        self._cam0_T_body = np.linalg.inv(self._body_T_cam0)

        grav_win = max(1, int(data.imu0_rate_hz * 0.5))
        first_cam_ts = stereo_matching_result.frames[first_keyframe_frame].timestamp_ns
        i0 = int(np.argmin(np.abs(imu_timestamps_ns - first_cam_ts)))
        gravity_in_body = self._imu_lin_accs[i0:i0 + grav_win].mean(axis=0)

        self._X = lambda i: gtsam.symbol('x', i)
        self._V = lambda i: gtsam.symbol('v', i)
        self._B = lambda i: gtsam.symbol('b', i)
        self._L = lambda i: gtsam.symbol('l', i)

        self._imu_params = gtsam.PreintegrationParams(gravity)
        self._imu_params.setGyroscopeCovariance(np.eye(3) * 1e-4)
        self._imu_params.setAccelerometerCovariance(np.eye(3) * 1e-3)
        self._imu_params.setIntegrationCovariance(np.eye(3) * 1e-8)

        self._PRIOR_POSE_NOISE = gtsam.noiseModel.Isotropic.Sigma(6, 0.1)
        self._PRIOR_VEL_NOISE = gtsam.noiseModel.Isotropic.Sigma(3, 0.1)
        self._PRIOR_BIAS_NOISE = gtsam.noiseModel.Isotropic.Sigma(6, 0.1)
        self._PNP_STEP_SIGMAS = np.array([0.01, 0.01, 0.01, 0.05, 0.05, 0.05])
        self._GYRO_BIAS_RW = 1.9393e-05
        self._ACCEL_BIAS_RW = 3.0000e-03
        self._FALLBACK_POSE_NOISE = gtsam.noiseModel.Isotropic.Sigma(6, 1.0)

        self._MIN_TRACK_LEN = 3
        self._PNP_FALLBACK_COVIS = 15
        PX_SIGMA = 1.5
        DEPTH_MIN, DEPTH_MAX = 0.3, 40.0
        self._cam0_K = gtsam.Cal3_S2(data.cam0_intrinsics.fx, data.cam0_intrinsics.fy, 0.0,
                                     data.cam0_intrinsics.cx, data.cam0_intrinsics.cy)
        self._cam1_K = gtsam.Cal3_S2(data.cam1_intrinsics.fx, data.cam1_intrinsics.fy, 0.0,
                                     data.cam1_intrinsics.cx, data.cam1_intrinsics.cy)
        self._cam0_pose = gtsam.Pose3(self._body_T_cam0)
        self._cam1_pose = gtsam.Pose3(data.cam1_extrinsics)
        self._PX_NOISE = gtsam.noiseModel.Robust.Create(
            gtsam.noiseModel.mEstimator.Huber.Create(1.345),
            gtsam.noiseModel.Isotropic.Sigma(2, PX_SIGMA))
        self._LM_PRIOR_NOISE = gtsam.noiseModel.Isotropic.Sigma(3, 5.0)

        self._track_builder = _LandmarkTrackBuilder(
            data, feature_detection_result, stereo_matching_result, optical_flow_result,
            DEPTH_MIN, DEPTH_MAX)
        self._tracks = self._track_builder.tracks
        self._loop_detector = (
            _LoopClosureDetector(data, feature_detection_result, stereo_matching_result)
            if enable_loop_closure else None)

        # node -> track ids observed there; and per-interval covisibility for the PnP gate. Keyed
        # by node index directly (not pre-sized by K, which isn't known until the run ends).
        self._nodes_to_tracks: dict[int, list[int]] = defaultdict(list)
        self._node_seen: dict[int, set[int]] = defaultdict(set)
        self._inserted_landmarks: set[int] = set()
        self._added_obs: set[tuple[int, int]] = set()
        self._n_proj_factors = 0
        # Cached per-node reprojection contributions -- see current_estimate's docstring for why
        # these are cached (and selectively refreshed) rather than rederived in full every call.
        self._sq_px = np.zeros(0)
        self._n_px = np.zeros(0)
        self._n_lm = np.zeros(0)

        self._isam2 = gtsam.ISAM2(gtsam.ISAM2Params())

        R_G_body0 = _align_vectors(gravity_in_body, -gravity)
        pose0 = gtsam.Pose3(gtsam.Rot3(R_G_body0), gtsam.Point3(0.0, 0.0, 0.0))
        f0, v0 = gtsam.NonlinearFactorGraph(), gtsam.Values()
        f0.add(gtsam.PriorFactorPose3(self._X(0), pose0, self._PRIOR_POSE_NOISE))
        f0.add(gtsam.PriorFactorVector(self._V(0), np.zeros(3), self._PRIOR_VEL_NOISE))
        f0.add(gtsam.PriorFactorConstantBias(self._B(0), gtsam.imuBias.ConstantBias(), self._PRIOR_BIAS_NOISE))
        v0.insert(self._X(0), pose0)
        v0.insert(self._V(0), np.zeros(3))
        v0.insert(self._B(0), gtsam.imuBias.ConstantBias())
        self._isam2.update(f0, v0)

        self._register_node_tracks(0, self._track_builder.add_keyframe(0, first_keyframe_frame))
        if self._loop_detector is not None:
            self._loop_detector.add_keyframe(0, first_keyframe_frame)  # never closes a cluster on its own

        self._j = 0  # index of the most recently added node
        self._prev_frame = first_keyframe_frame  # its original frame index

    def _register_node_tracks(self, node: int, touched: list[int]) -> None:
        self._nodes_to_tracks[node].extend(touched)
        self._node_seen[node].update(touched)

    def _add_obs_factors(self, factors: gtsam.NonlinearFactorGraph, tid: int, node: int,
                         uv0: np.ndarray, uv1: np.ndarray) -> None:
        factors.add(gtsam.GenericProjectionFactorCal3_S2(
            uv0, self._PX_NOISE, self._X(node), self._L(tid), self._cam0_K, False, False, self._cam0_pose))
        factors.add(gtsam.GenericProjectionFactorCal3_S2(
            uv1, self._PX_NOISE, self._X(node), self._L(tid), self._cam1_K, False, False, self._cam1_pose))
        self._added_obs.add((tid, node))
        self._n_proj_factors += 2

    def _process_node_landmarks(self, jj: int, est: gtsam.Values,
                                factors: gtsam.NonlinearFactorGraph, values: gtsam.Values) -> int:
        n_at_node = 0
        for tid in self._nodes_to_tracks[jj]:
            obs = self._tracks[tid]
            if tid in self._inserted_landmarks:
                if (tid, jj) not in self._added_obs:
                    o = next(o for o in obs if o.node == jj)
                    self._add_obs_factors(factors, tid, jj, o.uv0, o.uv1)
                    n_at_node += 1
                continue
            avail = [o for o in obs if o.node <= jj]
            if len(avail) < self._MIN_TRACK_LEN:
                continue
            first = obs[0]
            T_G_cam0 = est.atPose3(self._X(first.node)).matrix() @ self._body_T_cam0
            p_world = (T_G_cam0 @ np.append(first.point_cam0, 1.0))[:3]
            values.insert(self._L(tid), gtsam.Point3(*p_world))
            factors.add(gtsam.PriorFactorPoint3(self._L(tid), gtsam.Point3(*p_world), self._LM_PRIOR_NOISE))
            self._inserted_landmarks.add(tid)
            for o in avail:
                self._add_obs_factors(factors, tid, o.node, o.uv0, o.uv1)
                if o.node == jj:
                    n_at_node += 1
        return n_at_node

    def _add_loop_closure_edges(self, factors: gtsam.NonlinearFactorGraph, edges: list[_LoopClosureEdge]) -> None:
        for edge in edges:
            delta = _to_gtsam_pose3(edge.body_relative_pose)
            base_noise = gtsam.noiseModel.Diagonal.Sigmas(
                np.array([edge.rot_sigma] * 3 + [edge.trans_sigma] * 3))
            if edge.noise_mode == "huber":
                loop_noise = gtsam.noiseModel.Robust.Create(
                    gtsam.noiseModel.mEstimator.Huber.Create(1.345), base_noise)
            elif edge.noise_mode == "dcs":
                loop_noise = gtsam.noiseModel.Robust.Create(
                    gtsam.noiseModel.mEstimator.DCS.Create(6.0), base_noise)  # type: ignore[attr-defined]
            else:
                loop_noise = base_noise
            factors.add(gtsam.BetweenFactorPose3(self._X(edge.from_node), self._X(edge.to_node), delta, loop_noise))

    def add_keyframe(self, frame_idx: int) -> None:
        """Add a new node for `frame_idx` to the graph, linked to the previously-added node by an
        IMU factor (and a chained-PnP fallback when landmark covisibility isn't yet strong enough
        on its own) -- see _run_gtsam's _add_keyframe_to_graph for the algorithm this mirrors.
        """
        j = self._j
        est = self._isam2.calculateEstimate()
        pose_i = est.atPose3(self._X(j))
        vel_i = est.atVector(self._V(j))
        bias_i = est.atConstantBias(self._B(j))

        kf_i, kf_next = self._prev_frame, frame_idx
        kf_i_ts = self._sm_result.frames[kf_i].timestamp_ns
        kf_next_ts = self._sm_result.frames[kf_next].timestamp_ns

        self._register_node_tracks(j + 1, self._track_builder.add_keyframe(j + 1, frame_idx))
        new_loop_edges = (
            self._loop_detector.add_keyframe(j + 1, frame_idx) if self._loop_detector is not None else [])

        new_factors, new_values = gtsam.NonlinearFactorGraph(), gtsam.Values()

        pim = gtsam.PreintegratedImuMeasurements(self._imu_params, bias_i)
        window = np.where(
            (self._imu_timestamps_ns >= kf_i_ts) & (self._imu_timestamps_ns < kf_next_ts))[0]
        for k in window:
            dt = (float(self._imu_timestamps_ns[k + 1] - self._imu_timestamps_ns[k]) * 1e-9
                  if k + 1 < len(self._imu_timestamps_ns) else 1.0 / self._data.imu0_rate_hz)
            pim.integrateMeasurement(self._imu_lin_accs[k], self._imu_ang_vels[k], dt)

        new_factors.add(gtsam.ImuFactor(self._X(j), self._V(j), self._X(j + 1), self._V(j + 1), self._B(j), pim))
        dt_kf = float(kf_next_ts - kf_i_ts) * 1e-9
        bias_between_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array(
            [self._ACCEL_BIAS_RW * np.sqrt(dt_kf)] * 3 + [self._GYRO_BIAS_RW * np.sqrt(dt_kf)] * 3))
        new_factors.add(gtsam.BetweenFactorConstantBias(
            self._B(j), self._B(j + 1), gtsam.imuBias.ConstantBias(), bias_between_noise))

        nav_j = pim.predict(gtsam.NavState(pose_i, vel_i), bias_i)
        pose_init = nav_j.pose()
        vel_init = nav_j.velocity()

        strong_covis = sum(
            1 for tid in (self._node_seen[j] & self._node_seen[j + 1]) if tid in self._inserted_landmarks)
        pnp_added = False
        if strong_covis < self._PNP_FALLBACK_COVIS:
            pnp_cam0 = np.eye(4)
            pnp_ok = True
            for f in range(kf_i, kf_next):
                sm_f = self._sm_result.frames[f]
                fd_f, fd_f1 = self._fd_result.frames[f], self._fd_result.frames[f + 1]
                try:
                    rvec, tvec, _, _ = _run_pnp_step(
                        self._data, sm_f.points_3d, sm_f.matches,
                        fd_f.cam0_descriptors, fd_f1.cam0_keypoints, fd_f1.cam0_descriptors,
                    )
                except Exception:
                    pnp_ok = False
                    break
                pnp_cam0 = pnp_cam0 @ _rvec_tvec_to_transform(rvec, tvec)

            if pnp_ok:
                pnp_body = self._body_T_cam0 @ pnp_cam0 @ self._cam0_T_body
                pnp_delta = _to_gtsam_pose3(pnp_body)
                pnp_noise = gtsam.noiseModel.Diagonal.Sigmas(self._PNP_STEP_SIGMAS * np.sqrt(kf_next - kf_i))
                new_factors.add(gtsam.BetweenFactorPose3(self._X(j), self._X(j + 1), pnp_delta, pnp_noise))
                pose_init = pose_i.compose(pnp_delta)
                pnp_added = True

        new_values.insert(self._X(j + 1), pose_init)
        new_values.insert(self._V(j + 1), vel_init)
        new_values.insert(self._B(j + 1), bias_i)
        n_proj_at_next = self._process_node_landmarks(j + 1, est, new_factors, new_values)
        if not pnp_added and n_proj_at_next == 0:
            new_factors.add(gtsam.PriorFactorPose3(self._X(j + 1), pose_init, self._FALLBACK_POSE_NOISE))

        self._add_loop_closure_edges(new_factors, new_loop_edges)

        self._isam2.update(new_factors, new_values)
        self._j += 1
        self._prev_frame = frame_idx

    def _reprojection_for_node(
        self, node: int, final: gtsam.Values, inv_Twc0: list[np.ndarray], inv_Twc1: list[np.ndarray],
    ) -> tuple[float, float, float]:
        """sq_px/n_px/n_lm contributed by every inserted landmark observed at exactly this node,
        using the given (already current) camera-from-world transforms and landmark estimate --
        the per-node unit of work current_estimate's caching selectively redoes. Visits the same
        (landmark, observation) pairs the single pass over self._inserted_landmarks used to,
        just grouped by node instead of by landmark, via self._nodes_to_tracks (already maintained
        for the PnP covisibility gate) instead of re-scanning every landmark's whole history.
        """
        sq_px = 0.0
        n_px = 0.0
        n_lm = 0.0
        for tid in self._nodes_to_tracks[node]:
            if tid not in self._inserted_landmarks:
                continue
            obs = next((o for o in self._tracks[tid] if o.node == node), None)
            if obs is None:
                continue
            p = np.append(np.asarray(final.atPoint3(self._L(tid))), 1.0)
            seen = False
            for intrin, inv_Twc, uv in (
                (self._data.cam0_intrinsics, inv_Twc0, obs.uv0),
                (self._data.cam1_intrinsics, inv_Twc1, obs.uv1),
            ):
                pc = inv_Twc[node] @ p
                if pc[2] <= 1e-6:
                    continue
                u = intrin.fx * pc[0] / pc[2] + intrin.cx
                v = intrin.fy * pc[1] / pc[2] + intrin.cy
                sq_px += (u - uv[0]) ** 2 + (v - uv[1]) ** 2
                n_px += 1
                seen = True
            if seen:
                n_lm += 1.0
        return sq_px, n_px, n_lm

    def current_estimate(
        self, recompute_window: Optional[int] = None,
    ) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], np.ndarray, np.ndarray]:
        """Poses/velocities/biases/reprojection metrics for every node added so far. Unlike
        finalize(), doesn't flush any still-open loop-closure cluster (flushing early would
        prematurely close a cluster that a later keyframe might still have joined) -- safe to call
        after any keyframe, not just the last one, for a caller that wants to render a
        live-updating partial trajectory.

        recompute_window: if given, only the last `recompute_window` nodes' reprojection metrics
        are freshly recomputed from the current estimate on this call; every earlier node reuses
        whichever value was last computed for it (cached on self._sq_px/_n_px/_n_lm, grown as new
        nodes are added). This matters because ISAM2 keeps revising *past* poses/landmarks as new
        keyframes arrive, not just the newest ones -- measured drift of several centimeters on a
        settled node, converging within roughly 15-20 keyframes -- so a window trades a bounded
        amount of staleness in *old* keyframes' displayed reprojection RMSE during a live-updating
        run for turning this call from O(K) into O(recompute_window) (O(K) total across a run
        instead of O(K^2) -- this loop was 13.5% of one profiled run's total time, all of it spent
        re-deriving reprojection error for landmarks whose contribution had already converged; see
        tmp/slam_run_profile.html). None (the default) recomputes every node exactly, matching the
        original behavior bit-for-bit -- finalize() always uses this, so the *final* result is
        never affected, only intermediate live-preview snapshots (see _build_partial_result).
        """
        K = self._j + 1
        final = self._isam2.calculateEstimate()
        poses = [final.atPose3(self._X(j)).matrix() for j in range(K)]
        velocities = [final.atVector(self._V(j)) for j in range(K)]
        biases = [final.atConstantBias(self._B(j)).vector() for j in range(K)]

        inv_Twc0 = [np.linalg.inv(pm @ self._body_T_cam0) for pm in poses]
        inv_Twc1 = [np.linalg.inv(pm @ self._data.cam1_extrinsics) for pm in poses]

        if len(self._sq_px) < K:
            pad = K - len(self._sq_px)
            self._sq_px = np.concatenate([self._sq_px, np.zeros(pad)])
            self._n_px = np.concatenate([self._n_px, np.zeros(pad)])
            self._n_lm = np.concatenate([self._n_lm, np.zeros(pad)])

        recompute_from = 0 if recompute_window is None else max(0, K - recompute_window)
        for node in range(recompute_from, K):
            sq, n, lm = self._reprojection_for_node(node, final, inv_Twc0, inv_Twc1)
            self._sq_px[node] = sq
            self._n_px[node] = n
            self._n_lm[node] = lm

        n_px_K = self._n_px[:K]
        reprojection_rmse = np.where(n_px_K > 0, np.sqrt(self._sq_px[:K] / np.maximum(n_px_K, 1)), np.nan)

        return poses, velocities, biases, reprojection_rmse, self._n_lm[:K].copy()

    def finalize(self) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], np.ndarray, np.ndarray]:
        """Call once after every keyframe has been added: flushes any still-open loop-closure
        cluster, then returns current_estimate() -- see _run_gtsam's tail for the algorithm this
        mirrors.
        """
        if self._loop_detector is not None:
            final_edges = self._loop_detector.flush()
            if final_edges:
                tail_factors = gtsam.NonlinearFactorGraph()
                self._add_loop_closure_edges(tail_factors, final_edges)
                self._isam2.update(tail_factors, gtsam.Values())

        print(f"landmarks: {len(self._inserted_landmarks)}/{len(self._tracks)} tracks used, "
              f"{self._n_proj_factors} reprojection factors")
        # Every node, exactly -- see current_estimate's docstring on recompute_window=None.
        return self.current_estimate(recompute_window=None)


def _get_full_gt_arrays(data: EuRoCMAVData) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Ground-truth timestamps/positions/rotation-matrices for the *entire* dataset (every sample
    in data.ground_truth_samples, not just the run's window) -- meant to be computed once in
    _compute and passed into every _get_gtsam_result call, rather than each call rebuilding it
    from scratch (215x redundant reconstruction of a 29,993-sample array on one 60s/keyframe-215
    profiled run -- 14.9% of total run time; see tmp/slam_run_profile.html).

    Deliberately left unwindowed rather than narrowed to [min_timestamp_ns, max_timestamp_ns]:
    _get_gtsam_result searches this for the nearest ground-truth sample to each keyframe, and the
    true nearest sample to a keyframe right at the run's start/end boundary can legitimately fall
    just outside that window -- narrowing the search space would silently change which sample
    "nearest" finds there.
    """
    gt_timestamps_ns = np.array([s.timestamp_ns for s in data.ground_truth_samples])
    gt_positions = np.array([s.position for s in data.ground_truth_samples])
    gt_rotation_matrices = np.array([quaternion_to_rotation_matrix(s.quaternion) for s in data.ground_truth_samples])
    return gt_timestamps_ns, gt_positions, gt_rotation_matrices


def _get_gtsam_result(
    data: EuRoCMAVData,
    stereo_matching_result: StereoMatchingResult,
    first_timestamp_ns: int,
    min_timestamp_ns: int,
    gt_timestamps_ns: np.ndarray,
    gt_positions: np.ndarray,
    gt_rotation_matrices_all: np.ndarray,
    keyframe_indices: list[int],
    poses: list[np.ndarray],
    velocities: list[np.ndarray],
    biases: list[np.ndarray],
    reprojection_rmse: np.ndarray,
    landmark_counts: np.ndarray,
) -> SlamGtsamResult:
    """Ground-truth alignment and ATE/RPE metrics for an already-computed GTSAM trajectory.
    Doesn't run the graph itself -- see _compute, which drives _KeyframeScanner and
    _GtsamBuilder together in one pass and passes the resulting tuple straight in here, the same
    way _get_pnp_result already takes a pre-computed pnp_poses rather than deriving it itself.

    gt_timestamps_ns/gt_positions/gt_rotation_matrices_all come from _get_full_gt_arrays, computed
    once per run and shared across every call -- see that function's docstring for why they cover
    the whole dataset rather than just this run's window.
    """
    K = len(poses)

    kf_frames = [stereo_matching_result.frames[k] for k in keyframe_indices]
    cam_timestamps_ns = np.array([f.timestamp_ns for f in kf_frames])
    # Anchor to GT at the first GT sample inside the window (after start_s), consistent with
    # how the GT/PnP series are windowed.
    first_gt_sample = next(s for s in data.ground_truth_samples if s.timestamp_ns >= min_timestamp_ns)
    closest_cam_index = np.argmin(np.abs(cam_timestamps_ns - first_gt_sample.timestamp_ns))

    world_T_body_first = np.eye(4)
    world_T_body_first[:3, :3] = quaternion_to_rotation_matrix(first_gt_sample.quaternion)
    world_T_body_first[:3, 3] = first_gt_sample.position

    T_comp = world_T_body_first @ np.linalg.inv(poses[closest_cam_index])
    world_T_body_poses = np.array([T_comp @ T for T in poses])

    times = np.array([_elapsed_s(f.timestamp_ns, first_timestamp_ns) for f in kf_frames])

    # Per-keyframe position error vs the nearest ground-truth sample [m]. Poses are already
    # anchored to GT at closest_cam_index (T_comp), so this is a single-point-aligned error,
    # consistent with how positions are overlaid against GT in the view.
    nearest_gt = _nearest_sorted_indices(gt_timestamps_ns, cam_timestamps_ns)
    position_errors = np.linalg.norm(world_T_body_poses[:, :3, 3] - gt_positions[nearest_gt], axis=1)

    # ATE / RPE use the *raw*, un-anchored poses (before T_comp): T_comp is a full 6-DOF
    # transform derived from one GT sample's quaternion, so it can rotate the "up" axis away from
    # true gravity-up -- the raw poses are still exactly the gravity-aligned nav frame _run_gtsam
    # built, which _yaw_translation_align's roll/pitch-observable assumption depends on. RPE would
    # be unaffected either way (a constant transform cancels out of a relative pose), but ATE's
    # yaw-only alignment would not be consistent applied to the already-anchored poses.
    raw_positions = np.array([p[:3, 3] for p in poses])
    raw_rotation_matrices = np.array([p[:3, :3] for p in poses])
    gt_positions_at_kf = gt_positions[nearest_gt]
    gt_rotation_matrices_at_kf = gt_rotation_matrices_all[nearest_gt]
    ate_position_errors, ate_rotation_errors = _compute_ate(
        raw_positions, raw_rotation_matrices, gt_positions_at_kf, gt_rotation_matrices_at_kf)
    rpe_translation_errors, rpe_rotation_errors = _compute_rpe(
        times, raw_positions, raw_rotation_matrices, gt_positions_at_kf, gt_rotation_matrices_at_kf, RPE_DELTA_S)

    angular_velocities = []
    linear_accelerations = []
    velocities_np = np.array(velocities)
    for i in range(K - 1):
        dt = times[i + 1] - times[i]
        rotation_matrix = world_T_body_poses[i, :3, :3].T @ world_T_body_poses[i + 1, :3, :3]
        rotation_vector, _ = cv2.Rodrigues(rotation_matrix)
        angular_velocities.append(rotation_vector.flatten() / dt)

        acc_world = (velocities_np[i + 1] - velocities_np[i]) / dt
        acc_body = world_T_body_poses[i, :3, :3].T @ acc_world
        linear_accelerations.append(acc_body)

    rotation_matrices = world_T_body_poses[:, :3, :3]
    return SlamGtsamResult(
        times=times,
        positions=world_T_body_poses[:, :3, 3],
        attitudes=_mats_to_rvecs(rotation_matrices),
        rotation_matrices=rotation_matrices,
        velocities=velocities_np,
        biases=np.array(biases),
        position_errors=position_errors,
        ate_position_errors=ate_position_errors,
        ate_rotation_errors=ate_rotation_errors,
        rpe_translation_errors=rpe_translation_errors,
        rpe_rotation_errors=rpe_rotation_errors,
        reprojection_rmse=reprojection_rmse,
        landmark_counts=landmark_counts,
        angular_velocity_times=times[:-1],
        angular_velocities=np.array(angular_velocities),
        linear_accelerations=np.array(linear_accelerations),
    )


def _get_extra_result(
    data: EuRoCMAVData,
    gt_result: SlamGroundTruthResult,
    imu_result: SlamImuResult,
    min_timestamp_ns: int,
    max_timestamp_ns: int,
    gravity: np.ndarray,
) -> SlamExtraResult:

    gt_samples = [s for s in data.ground_truth_samples if min_timestamp_ns <= s.timestamp_ns <= max_timestamp_ns]
    gt_timestamps_ns = np.array([s.timestamp_ns for s in gt_samples])
    imu_timestamps_ns = np.array([s.timestamp_ns for s in data.imu_samples if min_timestamp_ns <= s.timestamp_ns <= max_timestamp_ns])
    closest_gt_indices = _nearest_sorted_indices(gt_timestamps_ns, imu_timestamps_ns)
    linear_accelerations_in_world = np.array([
        gt_result.rotation_matrices[idx] @ acc
        for idx, acc in zip(closest_gt_indices, imu_result.linear_accelerations)
    ])

    return SlamExtraResult(
        gravity=gravity,
        linear_accelerations_in_world=linear_accelerations_in_world,
    )


def _compute(
    data: EuRoCMAVData,
    start_s: float,
    duration_s: float,
    set_progress: Callable[[float, str], None],
    enable_loop_closure: bool = False,
    on_partial_result: Optional[Callable[[SlamResult], None]] = None,
) -> SlamResult:
    """Runs the whole SLAM stage -- feature detection, stereo matching, optical flow, keyframe
    selection, and GTSAM graph construction -- as one incremental pass over
    [start_s, start_s + duration_s], instead of requiring three fully-precomputed
    FeatureDetectionResult/StereoMatchingResult/OpticalFlowResult handed in from outside. Frame i
    is computed (_FrontendFrameComputer), then immediately offered to keyframe selection
    (_KeyframeScanner) and, for any keyframe it produces, the pose graph (_GtsamBuilder) -- before
    frame i+1 is ever touched. FeatureDetectionSolver/StereoMatchingSolver/OpticalFlowSolver still
    exist unchanged for their own diagnostic views (see tmp/fold_pipeline_into_slam_plan.html);
    this recomputes the same data independently rather than depending on their output.

    If on_partial_result is given, it's called with a full SlamResult (ground truth/IMU/extra
    unchanged from the final one -- none of those depend on the growing SLAM estimate -- but pnp
    and gtsam reflecting only what's been processed so far, via _KeyframeScanner.current_poses
    and _GtsamBuilder.current_estimate) after every keyframe, not just once at the end -- for a
    caller that wants to render the trajectory as it grows instead of only once the whole window
    has been processed.
    """
    first_timestamp_ns = data.cam_timestamps_ns[0]
    min_ts = first_timestamp_ns + int(start_s * 1e9)
    max_ts = min_ts + int(duration_s * 1e9)
    timestamps = [t for t in data.cam_timestamps_ns if min_ts <= t <= max_ts]
    N = len(timestamps)
    # SLAM runs on the frames sliced to the config's [start_s, start_s + duration_s] window, so
    # the first/last frame that actually exists in that range mark the window start/end -- not
    # min_ts/max_ts themselves, which may fall between two real frames. Trimming GT/IMU/extra to
    # the same bounds makes every series' time axis begin at start_s (times stay relative to the
    # dataset start), matching the PnP/GTSAM series that are already windowed.
    min_timestamp_ns = timestamps[0]
    max_timestamp_ns = timestamps[-1]

    gravity = np.array([0.0, 0.0, -9.81])

    # Progress budget across the whole solver so every step advances the bar:
    #   ground truth [0.00, 0.03]  IMU [0.03, 0.06]  frontend + keyframes + GTSAM [0.06, 0.96]
    #   extra/finishing [0.97, 1.00]
    set_progress(0.0, "Loading ground truth...")
    gt_result = _get_ground_truth_result(data, first_timestamp_ns, min_timestamp_ns, max_timestamp_ns)
    gt_timestamps_ns, gt_positions_all, gt_rotation_matrices_all = _get_full_gt_arrays(data)

    # Anchor the IMU-integrated orientation to GT at the first GT sample inside the window
    # (i.e. gt_result's first sample, which is gt_rotation_matrices[0]).
    set_progress(0.03, "Integrating IMU...")
    first_gt_timestamp_ns = next(
        s.timestamp_ns for s in data.ground_truth_samples if s.timestamp_ns >= min_timestamp_ns
    )
    imu_result = _get_imu_result(
        data, first_timestamp_ns, min_timestamp_ns, max_timestamp_ns,
        gt_result.rotation_matrices, first_gt_timestamp_ns,
    )
    # Depends only on gt_result/imu_result/gravity, all already available -- never on the growing
    # SLAM estimate -- so it's computed once here rather than after the loop below, to be ready
    # for on_partial_result's snapshots too (see _build_partial_result).
    extra_result = _get_extra_result(data, gt_result, imu_result, min_timestamp_ns, max_timestamp_ns, gravity)

    # Frame computation, keyframe selection, and GTSAM graph construction as one pass: each
    # frame's fd/sm/of is computed (_FrontendFrameComputer) and appended to these growing result
    # containers, then immediately offered to _KeyframeScanner and, for any keyframe it returns,
    # _GtsamBuilder -- before the next frame is ever touched. Both scanner/builder read from these
    # same containers (that's how they see earlier frames' data), so the containers are shared,
    # mutable state across the loop below, not a return value handed off afterward.
    set_progress(0.06, "Processing frames...")
    compute_t0 = time.monotonic()
    imu_samples = [s for s in data.imu_samples if min_timestamp_ns <= s.timestamp_ns <= max_timestamp_ns]

    frontend = _FrontendFrameComputer(data)
    feature_detection_result = FeatureDetectionResult(frames=[], elapsed_s=0.0)
    stereo_matching_result = StereoMatchingResult(frames=[], elapsed_s=0.0)
    optical_flow_result = OpticalFlowResult(frames=[], elapsed_s=0.0)

    def _process_frame(ts: int) -> None:
        fd_frame, sm_frame, of_frame = frontend.add_frame(ts)
        feature_detection_result.frames.append(fd_frame)
        stereo_matching_result.frames.append(sm_frame)
        optical_flow_result.frames.append(of_frame)

    def _build_partial_result(up_to_frame: int) -> SlamResult:
        """A full SlamResult reflecting only what's been processed through `up_to_frame` so far
        -- gt/imu/extra are the same final values computed above (none of them depend on the
        growing estimate); pnp and gtsam come from the scanner/builder's own live accessors.
        """
        partial_pnp_poses = scanner.current_poses(up_to_frame)
        partial_pnp_result = _get_pnp_result(
            data, stereo_matching_result, first_timestamp_ns, min_timestamp_ns, partial_pnp_poses)
        p_poses, p_vel, p_bias, p_rmse, p_lm = builder.current_estimate(
            recompute_window=_LIVE_PREVIEW_REPROJECTION_WINDOW)
        partial_gtsam_result = _get_gtsam_result(
            data, stereo_matching_result, first_timestamp_ns, min_timestamp_ns,
            gt_timestamps_ns, gt_positions_all, gt_rotation_matrices_all,
            scanner.keyframes, p_poses, p_vel, p_bias, p_rmse, p_lm,
        )
        return SlamResult(
            gt=gt_result, imu=imu_result, pnp=partial_pnp_result, gtsam=partial_gtsam_result,
            extra=extra_result,
        )

    _process_frame(timestamps[0])
    scanner = _KeyframeScanner(data, feature_detection_result, stereo_matching_result, N)
    builder = _GtsamBuilder(
        data, feature_detection_result, stereo_matching_result, optical_flow_result,
        imu_samples, gravity, first_keyframe_frame=0, enable_loop_closure=enable_loop_closure)
    for i in range(1, N):
        set_progress(0.06 + (i / N) * (0.95 - 0.06), "Processing frames...")
        _process_frame(timestamps[i])
        for kf in scanner.add_frame(i):
            builder.add_keyframe(kf)
            if on_partial_result is not None:
                on_partial_result(_build_partial_result(i))
    keyframe_indices = scanner.keyframes
    pnp_poses = scanner.finalize()
    gtsam_poses, velocities, biases, reprojection_rmse, landmark_counts = builder.finalize()
    # Frontend computation, scanning, and graph-building are now the same combined pass, so they
    # no longer have separately measurable costs -- both diagnostics report the same total.
    compute_elapsed = time.monotonic() - compute_t0

    pnp_result = _get_pnp_result(
        data, stereo_matching_result, first_timestamp_ns, min_timestamp_ns, pnp_poses)
    pnp_result.elapsed_time = compute_elapsed

    set_progress(0.96, "Aligning to ground truth...")
    gtsam_result = _get_gtsam_result(
        data, stereo_matching_result, first_timestamp_ns, min_timestamp_ns,
        gt_timestamps_ns, gt_positions_all, gt_rotation_matrices_all,
        keyframe_indices, gtsam_poses, velocities, biases, reprojection_rmse, landmark_counts,
    )
    gtsam_result.elapsed_time = compute_elapsed

    set_progress(0.97, "Finishing...")
    set_progress(1.0, "Done")
    return SlamResult(
        gt=gt_result,
        imu=imu_result,
        pnp=pnp_result,
        gtsam=gtsam_result,
        extra=extra_result,
    )


class _SolveCancelled(Exception):
    pass


class SlamSolver:
    """Runs the whole SLAM stage over [start_s, start_s + duration_s] -- feature detection,
    stereo matching, optical flow, keyframe selection, and GTSAM graph construction -- from raw
    EuRoCMAVData alone, the same (data, start_s, duration_s) shape FeatureDetectionSolver takes.
    Doesn't need a FeatureDetectionResult/StereoMatchingResult/OpticalFlowResult handed in: it
    recomputes that data itself, incrementally (see _compute), rather than depending on those
    solvers having already run. They still exist unchanged for their own diagnostic views.
    """

    def __init__(
        self, data: EuRoCMAVData, start_s: float, duration_s: float,
        cancel_event: Optional[threading.Event] = None,
        # Off by default: _LoopClosureDetector is O(K^2) brute-force descriptor matching (fine for
        # an offline research run, not yet acceptable as an always-on interactive-tool cost), and
        # its candidate-consolidation step is validated-but-not-tuned (see
        # _LoopClosureDetector's docstring) -- clear win on one regression-check sequence, real if
        # smaller localized regression on another. Opt in explicitly.
        enable_loop_closure: bool = False,
    ) -> None:
        self._data = data
        self._start_s = start_s
        self._duration_s = duration_s
        self._cancel_event = cancel_event if cancel_event is not None else threading.Event()
        self._enable_loop_closure = enable_loop_closure
        self.result: Optional[SlamResult] = None
        self.loading: bool = True
        self.error: Optional[str] = None
        self.progress: float = 0.0
        self.progress_label: str = ""

    def cancel(self) -> None:
        self._cancel_event.set()

    def run(self) -> None:
        def set_progress(value: float, label: str) -> None:
            if self._cancel_event.is_set():
                raise _SolveCancelled()
            self.progress = value
            self.progress_label = label

        def set_partial_result(result: SlamResult) -> None:
            if self._cancel_event.is_set():
                raise _SolveCancelled()
            # Same single-attribute-assignment pattern as set_progress above: safe to read from
            # the render thread without a lock (see SlamViewModel.start) since a reader either
            # sees the old SlamResult object or the new one, never a half-built one.
            self.result = result

        # Runs on the caller's background thread (see SlamViewModel.start), sharing this
        # process's memory: no spawn, no reimport, no pickling data across a process boundary.
        try:
            self.result = _compute(
                self._data, self._start_s, self._duration_s, set_progress,
                enable_loop_closure=self._enable_loop_closure, on_partial_result=set_partial_result)
        except _SolveCancelled:
            pass
        except Exception:
            self.error = traceback.format_exc()
        finally:
            self.loading = False
