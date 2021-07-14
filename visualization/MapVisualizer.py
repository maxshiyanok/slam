import typing as tp

import numpy as np
from plotly import graph_objects as go

from common.intrinsics import Intrinsics


class MapVisualizer:
    def __init__(self, fig: tp.Optional[go.Figure] = None) -> None:
        self._fig = fig if fig is not None else go.Figure()

    def __plot_camera(self, camera_pose: np.ndarray, intrinsics: Intrinsics, cam_num: int,
                      cam_size: float = 1e-2, rays: tp.Optional[np.ndarray] = None) -> None:
        R = camera_pose[:, :3]
        t = camera_pose[:, -1]
        H, W, F = np.array([intrinsics.cy, intrinsics.cx, intrinsics.fx]) * cam_size
        xyz = [[-W / 2, W / 2, W / 2, -W / 2, -W / 2, 0, W / 2, 0, W / 2, 0, -W / 2],
               [-H / 2, -H / 2, H / 2, H / 2, -H / 2, 0, -H / 2, 0, H / 2, 0, H / 2],
               [-F, -F, -F, -F, -F, 0, -F, 0, -F, 0, -F]]
        xyz = np.array(xyz)
        xyz[-1] /= 4
        xyz = R @ xyz + t[:, None]
        self._fig.add_trace(go.Mesh3d(x=xyz[0], y=xyz[1], z=xyz[2], opacity=0.5, color='rgb(0, 243, 0)',
                                      showlegend=False,
                                      name=f'cam_{cam_num}',
                                      legendgroup=f'cam_{cam_num}'))
        self._fig.add_trace(go.Scatter3d(x=xyz[0], y=xyz[1], z=xyz[2], line=dict(color='rgb(0, 0, 0)'), mode='lines',
                                         name=f'cam_{cam_num}',
                                         legendgroup=f'cam_{cam_num}'))
        if rays is not None:
            xyz = np.zeros((3, rays.shape[1] * 2))
            xyz[:, 1::2] = rays
            xyz = R @ xyz + t[:, None]
            self._fig.add_trace(go.Scatter3d(x=xyz[0], y=xyz[1], z=xyz[2], opacity=1,
                                             line=dict(color='rgb(255, 255, 255)'),
                                             mode='lines', showlegend=False, name='ray_trace'))

    def plot_cameras(self, poses: np.ndarray) -> None:
        # Works only for LLFF poses currently (axes are different between datasets)
        self._fig.add_trace(go.Cone(x=poses[:, 0, -1], y=-poses[:, 1, -1], z=poses[:, 2, -1],
                                    u=-poses[:, 0, 2], v=-poses[:, 1, 2], w=-poses[:, 2, 2]))
        self._fig.update_traces(text=list(map(str, np.arange(poses.shape[0]).tolist())), selector='cone')
        xyz_range = np.array([poses[..., -1].min(axis=0), poses[..., -1].max(axis=0)]).T
        self._fig.update_layout(width=800, height=600,
                                scene=dict(aspectratio=dict(x=1, y=1, z=1),
                                           xaxis=dict(range=xyz_range[0]),
                                           yaxis=dict(range=xyz_range[1]),
                                           zaxis=dict(range=xyz_range[2])))

    def plot_point_cloud(self, points: np.ndarray) -> None:
        self._fig.add_trace(go.Scatter3d(x=points[:, 0], y=points[:, 1], z=points[:, 2], name='Map points'))
        xyz_range = np.array([points.min(axis=0), points.max(axis=0)]).T
        self._fig.update_layout(width=800, height=600,
                                scene=dict(aspectratio=dict(x=1, y=1, z=1),
                                           xaxis=dict(range=xyz_range[0]),
                                           yaxis=dict(range=xyz_range[1]),
                                           zaxis=dict(range=xyz_range[2])))

    @property
    def figure(self) -> go.Figure:
        return self._fig


if __name__ == '__main__':
    viz = MapVisualizer()
    import pandas as pd
    from scipy.spatial.transform import Rotation
    scenario = 'llff_fern'
    poses = pd.read_csv(f'tests/{scenario}_slam_gt/ground_truth.txt', sep=' ', index_col='#frame_id')
    rot = Rotation.from_quat(poses.iloc[:, -4:]).as_matrix()
    poses = np.dstack((rot, poses.iloc[:, :3].to_numpy()[:, :, np.newaxis]))
    viz.plot_cameras(poses)
    viz.figure.show()
