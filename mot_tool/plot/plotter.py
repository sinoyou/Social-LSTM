import os
import pickle
import random

import numpy as np

import path_static
from mot_tool.plot.trajectory import Trajectory
from mot_tool.plot.trajectory_plot import TrajectoryPlot
from mot_tool.plot.util import get_scene_path


class Plotter:
    """
    Plotter类通过载入social_sample.py保存的pickle数据完成初始化，将每个行人的轨迹（真实或生成）以Trajectory类的形式存储起来。
    1.pickle数据中以每个行人为单元将数据组织在一个字典中：
     - dataset ： mot16的subname
     - real_traj ： (seq_length,5) 真实行人轨迹
     - generate_traj ：(seq_length,5) 生成行人轨迹
     - frame_seq ：（seq_length,1) 轨迹时间序列的真实FrameID
     - obs_length ：真实轨迹与生成轨迹中，已知和未知的breakpoint
     * real_traj中的数据在序列方向的头和尾存在无效数据（5个数均为0），这是因为DataLoader在截取固定片段时，行人还没进入或提前离开
    2.可视化功能支持
     - 给定ADE误差阈值和最小的有效预测长度，随机画一个轨迹
     - 批量导出所有轨迹至export_path文件夹
    """

    def __init__(self, pkl_file_path, export_dir):
        self.export_dir = export_dir
        self.trajectory_list = []

        # 加载raw文件
        path = pkl_file_path
        traj_f = open(path, 'rb')
        trajectory_list_raw = pickle.load(traj_f)
        # 整理成类
        for traj_raw in trajectory_list_raw:
            traj_process = Trajectory(traj_raw['dataset'], traj_raw['real_traj'], traj_raw['frame_seq'],
                                      traj_raw['obs_length'])
            traj_process.add_generate_trajectory(traj_raw['generate_traj'], 'Social LSTM with Scale')
            self.trajectory_list.append(traj_process)

    def plot_one_randomly(self, min_valid_pred_length=5, max_ade_error=0.10):
        # Choose one which read pred length is larger than 5
        cnt = 0
        # Get candidate trajectories
        traj_worth_show = []
        for traj in self.trajectory_list:
            # 查看轨迹的真实有效区间，分别是obs点前和后，只有预测的真实区间大于5的打印效果才比较好。
            valid_obs, valid_pred = traj.detect_valid_length()
            if valid_pred >= min_valid_pred_length and traj.is_worth_show(max_ade_error):
                traj_worth_show.append(traj)

        # plot on randomly
        random_index = random.randint(0, len(traj_worth_show) - 1)
        traj = traj_worth_show[random_index]
        tplot = self.__plot_one_trajectory__(traj)
        tplot.show()

    def export_all_trajectory(self, max_error=0.20, min_valid_pred_length=5):
        # 查看总导出文件夹是否存在
        if not os.path.exists(self.export_dir):
            os.mkdir(self.export_dir)

        # 开始导出
        for traj in self.trajectory_list:
            (valid_obs, valid_pred) = traj.detect_valid_length()  # 获取有效长度
            mot16_subname = traj.get_mot16_subname()  # 获取mot16的子数据库名称
            if traj.is_worth_show(max_error) and valid_pred >= min_valid_pred_length:
                export_sub_dir = os.path.join(self.export_dir, mot16_subname)
                # 查看子导出文件夹是否存在
                if not os.path.exists(export_sub_dir):
                    os.mkdir(export_sub_dir)
                tplot = self.__plot_one_trajectory__(traj)
                tplot.save(export_sub_dir)

    @staticmethod
    def __plot_one_trajectory__(traj: Trajectory):
        """
        根据轨迹类，在创建这个类的轨迹画板类，并返回这个类的应用
        :param traj: 轨迹类实例，其中包含一条gt轨迹和若干条生成的轨迹。
        """
        # 查看轨迹的真实有效区间，分别是obs点前和后，只有预测的真实区间大于5的打印效果才比较好。
        valid_obs, valid_pred = traj.detect_valid_length()
        obs_length = traj.get_obs_length()
        # 获取真实轨迹 (valid_pred x 5)
        ground_truth_traj = traj.get_real_trajectory()[obs_length:obs_length + valid_pred, :]
        # 获取背景图片
        img_path = get_scene_path(os.path.join('..', '..', 'data', 'MOT16Full'), traj)
        # 创建plot类
        total_loss, counter = traj.calculate_total_loss()
        ave_loss = 'NaN' if counter == 0 else total_loss / counter
        name = 'MOT16-{}_pedID={:d}_frameID={}_obs={}_pred={}_loss={}'.format(traj.get_mot16_subname(),
                                                                              int(ground_truth_traj[0, 0]),
                                                                              traj.get_pred_start_frame(),
                                                                              valid_obs, valid_pred, ave_loss)
        tplot = TrajectoryPlot(img_path=img_path, box_cor=np.squeeze(ground_truth_traj[0, 1:5]), name=name, squeeze=1)

        # 添加轨迹
        # for ground truth trajectory
        gt_traj = traj.get_real_trajectory()[obs_length:obs_length + valid_pred, :]
        tplot.add_one_trajectory(gt_traj[:, 1], gt_traj[:, 2], 'Ground Truth')
        # for generated trajectory
        gen_traj_dict = traj.get_all_generate_trajectories()
        for (name, value) in gen_traj_dict.items():
            gen_traj = value[obs_length:obs_length + valid_pred, :]
            tplot.add_one_trajectory(gen_traj[:, 1], gen_traj[:, 2], name)
        return tplot


if __name__ == '__main__':
    # 存储pickle文件的路径
    pkl_file_dir = path_static.save_path
    pkl_file_path = os.path.join('..', '..', pkl_file_dir, 'traj_file_raw')
    # 相关资料保存的路径
    save_path = os.path.join('..', '..', pkl_file_dir, 'trajectory_plot')
    plotter = Plotter(pkl_file_path, save_path)
    plotter.export_all_trajectory()
