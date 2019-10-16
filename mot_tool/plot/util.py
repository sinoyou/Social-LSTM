import os

from mot_tool.plot.trajectory import Trajectory


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
    path = os.path.join(dir, 'train', 'MOT16-' + database, 'img1', img_name)
    return path
