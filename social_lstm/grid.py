'''
Helper functions to compute the masks relevant to social grid

Author : Anirudh Vemula
Date : 29th October 2016
'''
import numpy as np


def getGridMask(frame, neighborhood_size, grid_size):
    '''
    This function computes the binary mask that represents the
    occupancy of each ped in the other's grid
    params:
    frame : This will be a MNP x 5 matrix with each row being [pedID, x, y, scale-x, scale-y]
    neighborhood_size : Scalar value representing the size of neighborhood considered
    grid_size : Scalar value representing the size of the grid discretization
    '''

    # Maximum number of pedestrians
    mnp = frame.shape[0]
    frame_mask = np.zeros((mnp, mnp, grid_size ** 2))

    # width_bound, height_bound = neighborhood_size / (width * 1.0), neighborhood_size / (height * 1.0)
    width_bound, height_bound = neighborhood_size, neighborhood_size

    # For each ped in the frame (existent and non-existent)
    for pedindex in range(mnp):
        # If pedID is zero, then non-existent ped
        if frame[pedindex, 0] == 0:
            # Binary mask should be zero for non-existent ped
            continue

        # Get x and y of the current ped
        current_x, current_y = frame[pedindex, 1], frame[pedindex, 2]

        width_low, width_high = current_x - width_bound / 2, current_x + width_bound / 2
        height_low, height_high = current_y - height_bound / 2, current_y + height_bound / 2

        # For all the other peds
        for otherpedindex in range(mnp):
            # If other pedID is zero, then non-existent ped
            if frame[otherpedindex, 0] == 0:
                # Binary mask should be zero
                continue

            # If the other pedID is the same as current pedID
            if frame[otherpedindex, 0] == frame[pedindex, 0]:
                # The ped cannot be counted in his own grid
                continue

            # Get x and y of the other ped
            other_x, other_y = frame[otherpedindex, 1], frame[otherpedindex, 2]
            if other_x >= width_high or other_x < width_low or other_y >= height_high or other_y < height_low:
                # Ped not in surrounding, so binary mask should be zero
                continue

            # If in surrounding, calculate the grid cell
            cell_x = int(np.floor(((other_x - width_low) / width_bound) * grid_size))
            cell_y = int(np.floor(((other_y - height_low) / height_bound) * grid_size))
            # 浮点数精度问题，可能存在越界的可能
            cell_x = min(1, cell_x)
            cell_y = min(1, cell_y)
            # Other ped is in the corresponding grid cell of current ped
            frame_mask[pedindex, otherpedindex, cell_x + cell_y * grid_size] = 1

    return frame_mask


def getSequenceGridMask(sequence, neighborhood_size, grid_size):
    '''
    Get the grid masks for all the frames in the sequence
    params:
    sequence : A numpy matrix of shape SL x MNP x 5
    neighborhood_size : Scalar value representing the size of neighborhood considered
    grid_size : Scalar value representing the size of the grid discretization
    '''
    # yzn ：形成Mask有两步要走：
    # 第一步 - 查看边界条件，确定某个行人是否需要出现在行人A的Mask中。(主要涉及dimensions和neighborhood size)
    # 第二步 - 若在行人A的Mask中，该在Mask的哪一个部位？(主要涉及grid_size)
    # * dimensions列表中的两个值和neighborhood size隶属同一个比例尺，neighborhood size就是环境的极限大小。（可理解为图片边界）
    #   neighborhood size用于在dimensions全局环境下划分的grid边界，相除区间处于0-1，恰好符合输入数据的规格要求。
    sl = sequence.shape[0]
    mnp = sequence.shape[1]
    sequence_mask = np.zeros((sl, mnp, mnp, grid_size ** 2))

    for i in range(sl):
        sequence_mask[i, :, :, :] = getGridMask(sequence[i, :, :], neighborhood_size, grid_size)

    return sequence_mask
