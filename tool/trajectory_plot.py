import os

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import cv2


class Trajectory:
    """
    轨迹类
    """

    def __init__(self, dataset, real_traj, frames, obs_length, mean_error, final_error):
        self.dataset = dataset
        self.real_traj = real_traj.copy()
        self.frames = frames.copy()
        self.obs_length = obs_length
        self.mean_error = mean_error
        self.final_error = final_error

        self.gen_traj = {}

    def get_pred_start_frame(self):
        return self.frames[self.obs_length]

    def get_mot16_subname(self):
        return self.dataset

    def is_worth_show(self):
        return False

    def get_real_trajectory(self):
        return self.real_traj[self.obs_length:]

    def get_all_generate_trajectories(self):
        return self.gen_traj

    def add_generate_trajectory(self, gen_traj, name):
        self.gen_traj[name] = gen_traj.copy()


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
        self.traj_list.append((x_array, y_array))

    def plot(self):
        fig, ax = plt.subplots()
        img_width = self.width
        img_height = self.height

        # put image on the plot at first
        ax.imshow(self.img)

        # put bounding box
        if self.box_cor:
            (center_x, center_y, box_width, box_height) = self.box_cor
            top_left_x = center_x - box_width / 2
            top_left_y = center_y - box_height / 2
            xy_real = (top_left_x * img_width, top_left_y * img_height)
            xy_scale = (box_width * img_width, box_height * img_height)
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


def get_scene_path(dir, trajectory: Trajectory):
    """
    给定数据目录和轨迹类，返回背景图片的确定地址
    :param dir:
    :param trajectory:
    :return:
    """
    pred_start_frame = trajectory.get_pred_start_frame()
    img_name = '{:0>6d}.jpg'.format(pred_start_frame)
    database = trajectory.get_mot16_subname()
    path = os.path.join(dir, 'MOT16-', database, 'img1', img_name)
    return path


if __name__ == '__main__':
    tp = TrajectoryPlot('sample.jpg', (0.5, 0.5, 0.1, 0.2), squeeze=1)
    tp.add_one_trajectory(np.array([0.1, 0.3, 0.4]), np.array([0.2, 0.1, 0.3]))
    tp.show()
