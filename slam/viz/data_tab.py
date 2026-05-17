import numpy as np
from imgui_bundle import imgui

from slam import DataFolder


def _matrix_table(m: np.ndarray) -> None:
    rows, cols = m.shape
    if imgui.begin_table("##mat", cols, imgui.TableFlags_.borders | imgui.TableFlags_.sizing_fixed_fit):
        for r in range(rows):
            imgui.table_next_row()
            for c in range(cols):
                imgui.table_next_column()
                imgui.text(f"{m[r, c]:.6f}")
        imgui.end_table()


def data_tab(data: DataFolder) -> None:
    imgui.begin_child("##data_scroll", (0, 0), False)

    imgui.text(f"path: {data.path}")
    imgui.separator()

    imgui.text(f"cam_timestamps_ns ({len(data.cam_timestamps_ns)} total):")
    for i, ts in enumerate(data.cam_timestamps_ns[:5]):
        imgui.text(f"  [{i}] {ts}")
    imgui.separator()

    imgui.text(f"imu_samples ({len(data.imu_samples)} total):")
    for i, s in enumerate(data.imu_samples[:5]):
        imgui.text(f"  [{i}] t={s.timestamp_ns}  ang_vel={s.angular_velocity}  lin_acc={s.linear_acceleration}")
    imgui.separator()

    imgui.text(f"ground_truth_samples ({len(data.ground_truth_samples)} total):")
    for i, s in enumerate(data.ground_truth_samples[:5]):
        imgui.text(f"  [{i}] t={s.timestamp_ns}  pos={s.position}  quat={s.quaternion}")
    imgui.separator()

    imgui.text(f"leica_samples ({len(data.leica_samples)} total):")
    for i, s in enumerate(data.leica_samples[:5]):
        imgui.text(f"  [{i}] t={s.timestamp_ns}  pos={s.position}")
    imgui.separator()

    imgui.text("cam0_extrinsics:")
    _matrix_table(data.cam0_extrinsics)

    imgui.text("cam1_extrinsics:")
    _matrix_table(data.cam1_extrinsics)

    imgui.text("leica_extrinsics:")
    _matrix_table(data.leica_extrinsics)
    imgui.separator()

    for name, intr in [("cam0_intrinsics", data.cam0_intrinsics), ("cam1_intrinsics", data.cam1_intrinsics)]:
        imgui.text(f"{name}:")
        imgui.text(f"  fx={intr.fx}  fy={intr.fy}  cx={intr.cx}  cy={intr.cy}")
        imgui.text(f"  k1={intr.k1}  k2={intr.k2}  p1={intr.p1}  p2={intr.p2}")
        imgui.separator()

    imgui.end_child()
