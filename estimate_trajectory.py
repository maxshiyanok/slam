from common.dataset import Dataset
from common.trajectory import Trajectory


def estimate_trajectory(data_dir, out_dir):
    # TODO: fill trajectory here
    trajectory = {}
    Trajectory.write(Dataset.get_result_poses_file(out_dir), trajectory)
