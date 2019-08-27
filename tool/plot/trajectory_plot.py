import os
import pickle

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import cv2
import path_static
from tool.plot.trajectory import Trajectory
from tool.plot.util import get_scene_path


class TrajectoryPlot:
    def __init__(self, img_path, box_cor, squeeze=1.0):
        self.num_trajectory = 0
        self.ptr_color = 0
        self.des_list = []
        self.traj_list = []
        self.box_cor = box_cor
        self.subplot = plt.subplot()
        self.color_plant = ['#1C86EE', '#B3EE3A', '#CD00CD', '#EE0000', '#BFEFFF']

        # load image as background and squeeze it.
        ori_img = cv2.imread(img_path)
        height = ori_img.shape[0]
        width = ori_img.shape[1]
        self.img = cv2.resize(ori_img, dsize=(int(width * squeeze), int(height * squeeze)))
        self.width = width * squeeze
        self.height = height * squeeze

    def add_one_trajectory(self, x_array, y_array, description=None):
        self.num_trajectory = self.num_trajectory + 1
        if not description:
            self.des_list.append('line-' + str(self.num_trajectory))
        else:
            self.des_list.append(description)
        self.traj_list.append((x_array, y_array))

    def plot(self):
        fig, ax = plt.subplots()
        img_width = self.width
        img_height = self.height

        # put image on the plot at first
        ax.imshow(self.img)

        # put bounding box
        if self.box_cor.any():
            (center_x, center_y, box_width, box_height) = self.box_cor
            top_left_x = center_x - box_width
            top_left_y = center_y - box_height
            xy_real = (top_left_x * img_width, top_left_y * img_height)
            xy_scale = (box_width * img_width * 2, box_height * img_height * 2)
            print("Real-Scale", xy_real, xy_scale)
            rect = mpatches.Rectangle(xy_real, xy_scale[0], xy_scale[1], color='r', fill=False)
            ax.add_patch(rect)

        # put trajectories
        for i, traj in enumerate(self.traj_list):
            description = self.des_list[i]
            x_array, y_array = self.traj_list[i]
            x_array, y_array = x_array * img_width, y_array * img_height  # revert to real img size.

            # 根据尺寸和环境获取合适的线条参数
            color = self.__get_color__()
            line_width = self.__get_line_width__()
            marker_size = self.__get_marker_size__()

            ax.plot(x_array, y_array, '--', linewidth=line_width, color=color, label=description)
            ax.plot(x_array[-1], y_array[-1], 'x', markersize=marker_size, color=color)

        # put legend
        plt.legend()

    def save(self, save_path):
        self.plot()
        pass

    def show(self):
        self.plot()
        plt.show()

    def __get_color__(self):
        if self.ptr_color < len(self.color_plant):
            self.ptr_color = self.ptr_color + 1
            return self.color_plant[self.ptr_color - 1]
        else:
            raise Exception('NoEnoughColors')

    @staticmethod
    def __get_line_width__():
        # min_scale = min(self.height, self.width)
        # return round(min_scale / 100)
        return 2

    @staticmethod
    def __get_marker_size__():
        # min_scale = min(self.height, self.width)
        # return round(min_scale / 20)
        return 10


if __name__ == '__main__':
    # tp = TrajectoryPlot('sample.jpg', (0.5, 0.5, 0.1, 0.2), squeeze=1)
    # tp.add_one_trajectory(np.array([0.1, 0.3, 0.4]), np.array([0.2, 0.1, 0.3]))
    # tp.show()
    save_dir = path_static.save_path
    path = os.path.join('..', '..', save_dir, 'traj_file_raw')
    traj_f = open(path, 'rb')
    trajectory_list_raw = pickle.load(traj_f)
    trajectory_list = []
    # 整理成类
    for traj_raw in trajectory_list_raw:
        traj_process = Trajectory(traj_raw['dataset'], traj_raw['real_traj'], traj_raw['frame_seq'],
                                  traj_raw['obs_length'])
        traj_process.add_generate_trajectory(traj_raw['generate_traj'], 'Social LSTM with Scale')
        trajectory_list.append(traj_process)
    # Choose one which read pred length is larger than 5
    cnt = 0
    for traj in trajectory_list:
        # 查看轨迹的真实有效区间，分别是obs点前和后，只有预测的真实区间大于5的打印效果才比较好。
        valid_obs, valid_pred = traj.detect_valid_length()
        if valid_pred >= 5 and traj.is_worth_show():
            obs_length = traj.get_obs_length()
            # 获取真实轨迹 (valid_pred x 5)
            ground_truth_traj = traj.get_real_trajectory()[obs_length:obs_length + valid_pred, :]
            # 获取背景图片
            img_path = get_scene_path(os.path.join('..', '..', 'data', 'MOT16Full'), traj)
            # 创建plot类
            tplot = TrajectoryPlot(img_path, np.squeeze(ground_truth_traj[0, 1:5]), 1)
            # 添加轨迹
            # for ground truth trajectory
            gt_traj = traj.get_real_trajectory()[obs_length:obs_length + valid_obs, :]
            tplot.add_one_trajectory(gt_traj[:, 1], gt_traj[:, 2], 'Ground Truth')
            # for generated trajectory
            gen_traj_dict = traj.get_all_generate_trajectories()
            for (name, value) in gen_traj_dict.items():
                gen_traj = value[obs_length:obs_length + valid_obs, :]
                tplot.add_one_trajectory(gen_traj[:, 1], gen_traj[:, 2], name)
            pass
            # show
            print('MOT16-{}, pedID={}, frameID={}'.format(traj.get_mot16_subname(), ground_truth_traj[0, 0],
                                                          traj.get_pred_start_frame()))
            print('Coordinate:', ground_truth_traj[0, 1:5])
            tplot.show()
            break
