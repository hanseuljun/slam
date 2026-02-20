import matplotlib.pyplot as plt
import numpy as np


def plot_positions(
    series: list[tuple[np.ndarray, np.ndarray, str]],
):
    fig, (ax_x, ax_y, ax_z) = plt.subplots(1, 3, figsize=(12, 4))
    fig.suptitle('Position')

    labels = ['X', 'Y', 'Z']
    for ax, i in zip([ax_x, ax_y, ax_z], range(3)):
        for times, positions, label in series:
            ax.plot(times, positions[:, i], label=label)
        ax.set_xlabel('Time [s]')
        ax.set_ylabel(f'{labels[i]} [m]')
        ax.legend()

    plt.tight_layout()
    plt.show()


def plot_angular_velocities(
    series: list[tuple[np.ndarray, np.ndarray, str]],
):
    fig, (ax_wx, ax_wy, ax_wz) = plt.subplots(3, 1, figsize=(12, 9))
    fig.suptitle('Angular Velocity in Body Frame')

    labels = ['wx', 'wy', 'wz']
    for ax, i in zip([ax_wx, ax_wy, ax_wz], range(3)):
        for times, angular_velocities, label in series:
            ax.plot(times, angular_velocities[:, i], label=label)
        ax.set_xlabel('Time [s]')
        ax.set_ylabel(f'{labels[i]} [rad/s]')
        ax.legend()

    plt.tight_layout()
    plt.show()


def plot_rotation_axes(
    series: list[tuple[np.ndarray, np.ndarray, str]],
):
    fig, axes = plt.subplots(3, 3, figsize=(12, 9))
    fig.suptitle('Rotation Axes')

    axis_names = ['Right (x-axis)', 'Up (y-axis)', 'Forward (z-axis)']
    component_names = ['X', 'Y', 'Z']

    for row in range(3):
        for col in range(3):
            ax = axes[row, col]
            for times, attitudes, label in series:
                ax.plot(times, attitudes[:, col, row], label=label)
            ax.set_xlabel('Time [s]')
            ax.set_ylabel(f'{axis_names[row]} {component_names[col]}')
            ax.legend()

    plt.tight_layout()
    plt.show()
