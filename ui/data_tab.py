import numpy as np
from nicegui import ui

from slam import DataFolder


def _matrix_table(m: np.ndarray) -> None:
    rows, cols = m.shape
    html = '<table style="border-collapse:collapse;">'
    for r in range(rows):
        html += '<tr>'
        for c in range(cols):
            html += f'<td style="padding:4px;border:1px solid #ccc;">{m[r, c]:.6f}</td>'
        html += '</tr>'
    html += '</table>'
    ui.html(html)


def data_tab(data: DataFolder) -> None:
    with ui.scroll_area().classes('w-full h-full'):
        ui.label(f'path: {data.path}')
        ui.separator()

        ui.label(f'cam_timestamps_ns ({len(data.cam_timestamps_ns)} total):')
        for i, ts in enumerate(data.cam_timestamps_ns[:5]):
            ui.label(f'  [{i}] {ts}')
        ui.separator()

        ui.label(f'imu_samples ({len(data.imu_samples)} total):')
        for i, s in enumerate(data.imu_samples[:5]):
            ui.label(f'  [{i}] t={s.timestamp_ns}  ang_vel={s.angular_velocity}  lin_acc={s.linear_acceleration}')
        ui.separator()

        ui.label(f'ground_truth_samples ({len(data.ground_truth_samples)} total):')
        for i, s in enumerate(data.ground_truth_samples[:5]):
            ui.label(f'  [{i}] t={s.timestamp_ns}  pos={s.position}  quat={s.quaternion}')
        ui.separator()

        ui.label(f'leica_samples ({len(data.leica_samples)} total):')
        for i, s in enumerate(data.leica_samples[:5]):
            ui.label(f'  [{i}] t={s.timestamp_ns}  pos={s.position}')
        ui.separator()

        ui.label('cam0_extrinsics:')
        _matrix_table(data.cam0_extrinsics)
        ui.label('cam1_extrinsics:')
        _matrix_table(data.cam1_extrinsics)
        ui.label('leica_extrinsics:')
        _matrix_table(data.leica_extrinsics)
        ui.separator()

        for name, intr in [('cam0_intrinsics', data.cam0_intrinsics), ('cam1_intrinsics', data.cam1_intrinsics)]:
            ui.label(f'{name}:')
            ui.label(f'  fx={intr.fx}  fy={intr.fy}  cx={intr.cx}  cy={intr.cy}')
            ui.label(f'  k1={intr.k1}  k2={intr.k2}  p1={intr.p1}  p2={intr.p2}')
            ui.separator()
