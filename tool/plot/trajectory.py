import numpy as np


class Trajectory:
    """
    轨迹类
    """

    def __init__(self, dataset, real_traj, frame_seq, obs_length):
        """

        :param dataset: dataset subname in mot16 dataset.
        :param real_traj: shape [seq_length, 5]. 5 = pedID, x, y, scale-x, scale-y.
        :param frame_seq: shape [seq_length, 1]. Contains real frameID corresponding to sequence.
        :param obs_length: A break point. Before the point is sequence data put into model, After(or equal) the point
        is sequence the model generated.
        """
        self.dataset = dataset
        self.real_traj = real_traj.copy()
        self.frame_seq = frame_seq.copy()
        self.obs_length = obs_length

        self.gen_traj = {}

    def get_pred_start_frame(self):
        return int(self.frame_seq[self.obs_length, 0])

    def get_mot16_subname(self):
        return self.dataset

    def is_worth_show(self):
        """
        根据某些指标（如误差）判断该轨迹是否值得展示
        :return:
        """
        counter = 0
        loss = 0
        valid_obs, valid_pred = self.detect_valid_length()
        for (key, value) in self.gen_traj.items():
            gen_traj = value
            for j in range(self.obs_length + valid_pred - 1, self.real_traj.shape[0]):
                if self.real_traj[j][0] != 0:
                    counter += 1
                    loss += np.linalg.norm(self.real_traj[j, [1, 2]] - gen_traj[j, [1, 2]])

        ave_loss = loss / counter
        if ave_loss < 0.03:
            return True
        else:
            return False

    def get_real_trajectory(self):
        """
        获取完整的真实路径（ground truth)，其中大概率包含无效的区间（分别处于头和尾）。
        :return:
        """
        return self.real_traj

    def get_all_generate_trajectories(self):
        return self.gen_traj

    def get_obs_length(self):
        return self.obs_length

    def add_generate_trajectory(self, gen_traj, name):
        self.gen_traj[name] = gen_traj.copy()

    def detect_valid_length(self):
        """
        根据真实路径中行人id的分布，返回真实路径obs_len前后段的真正有效部分。
        :return: a tuple (obs_valid_length, pred_valid_length)
        obs_valid_length >= 0. pred_valid_length >= 0
        """
        obs_valid_length = 0
        for i in range(self.obs_length - 1, -1, -1):
            if self.real_traj[i][0] != 0:
                obs_valid_length += 1
            else:
                break

        pred_valid_length = 0
        for i in range(self.obs_length, self.real_traj.shape[0], 1):
            if self.real_traj[i][0] != 0:
                pred_valid_length += 1
            else:
                break
        return obs_valid_length, pred_valid_length
