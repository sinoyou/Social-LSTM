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
    def __init__(self, img_path, box_cor, name, squeeze=1.0):
        """
        初始化一个行人轨迹的画板
        :param img_path: 背景图片
        :param box_cor: 行人框属性 - center-x,center-y,width/2,height/2
        :param name: 画板名称
        :param squeeze: 背景图片压缩比例
        """
        self.num_trajectory = 0
        self.ptr_color = 0
        self.des_list = []
        self.traj_list = []
        self.box_cor = box_cor
        self.subplot = plt.subplot()
        self.color_plant = ['#1C86EE', '#B3EE3A', '#CD00CD', '#EE0000', '#BFEFFF']
        self.name = name

        # load image as background and squeeze it.
        ori_img = cv2.imread(img_path)
        height = ori_img.shape[0]
        width = ori_img.shape[1]
        self.img = cv2.resize(ori_img, dsize=(int(width * squeeze), int(height * squeeze)))
        self.width = width * squeeze
        self.height = height * squeeze

    def add_one_trajectory(self, x_array, y_array, description=None):
        """
        向画板中增加一条轨迹
        """
        self.num_trajectory = self.num_trajectory + 1
        if not description:
            self.des_list.append('line-' + str(self.num_trajectory))
        else:
            self.des_list.append(description)
        self.traj_list.append((x_array, y_array))

    def add_name(self, name):
        self.name = name

    def save(self, save_dir=None):
        if not os.path.exists(save_dir):
            print('[Error]: Save directory not exist.')
        else:
            # 画图
            self.__plot__()
            path = os.path.join(save_dir, self.name + '.jpg')
            plt.savefig(path)
            print('Save Plot {} Succeed.'.format(self.name))
            self.__close_plot__()

    def show(self):
        self.__plot__()
        plt.show()
        self.__close_plot__()

    def __plot__(self):
        """
        根据行人框和轨迹数据，在画板上生成。
        :return:
        """
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
        # put name
        plt.title(self.name)

    def __close_plot__(self):
        """
        操作与self.plot相反，用于显示或导出后清除画板释放内存
        :return:
        """
        plt.close()

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
    tp = TrajectoryPlot(img_path='sample.jpg', box_cor=np.array([0.5, 0.5, 0.1, 0.2]), squeeze=1, name='sample')
    tp.add_one_trajectory(np.array([0.1, 0.3, 0.4]), np.array([0.2, 0.1, 0.3]))
    tp.show()
