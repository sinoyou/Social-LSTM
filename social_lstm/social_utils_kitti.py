import os
import pickle
import string

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


class KittiDataLoader():
    def __init__(self, batch_size, seq_length, max_num_peds, ignore_list, sub_set, database_dir='data/KITTI'):
        """
        DataLoader for <label> data in KITTI dataset: "multi-object tracking 2012"
        Download Addr: http://www.cvlibs.net/download.php?file=data_tracking_label_2.zip
        Ref:
        [1] K. Bernardin, R. Stiefelhagen:
            Evaluating Multiple Object Tracking Performance: The CLEAR MOT Metrics. JIVP 2008.
        :param batch_size: batch size.
        :param seq_length: sequence length. assume obs_sequence & pred_sequence share same length,
                           seq_length-1 of them are same. If you want to load data with two sequences not public part,
                           please make seq_length = obs_length + pred_length and divide returned value manually.
        :param max_num_peds: maximum number of pedestrians in one scene.
        :param ignore_list: ignored scenes in Kitti Database.
        :param sub_set: train, *val, test
        """
        # save parameters
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.max_num_peds = max_num_peds
        self.ignore_list = ignore_list
        self.database_dir = database_dir
        self.sub_set = sub_set

        # pre process data
        list_preprocess_data = self.preprocess()

        # select valid processed sub-sequence of scene with seq_length larger than seq_length
        self.list_sub_scene_data = [x for x in list_preprocess_data if
                                    x['data'].shape[0] >= self.seq_length + 1]

        # calculate data's attributes.
        self.num_sequence = 0
        self.sub_scene_ptr = 0
        self.sequence_ptr = 0
        for data in self.list_sub_scene_data:
            self.num_sequence += data['data'].shape[0] - (self.seq_length + 1) + 1

        print('Batch size = {}, Batches = {}'.format(self.batch_size, self.__len__()))

    def preprocess(self):
        """
        Pre process *.txt data to list of sub-scene matrix.
        * Here, sub-scene refers to partial sequence in a whole scene,
          we do this because the data is raw and target objects(pedestrians + [cyclist])
          do not appear in all the scene. Many slice (which matrix all zero) should be ignored.
        :return: list of matrix
        """
        # load data -> preprocess_files_candidate
        label_files_path = os.path.join(self.database_dir, self.sub_set)
        label_files = os.listdir(label_files_path)
        names = ['frame', 'track id', 'type', 'truncated', 'occluded', 'alpha',
                 'bbox_left', 'bbox_top', 'bbox_right', 'bbox_bottom',
                 'dimensions_x', 'dimension_y', 'dimension_z',
                 'location_x', 'location_y', "location_z",
                 'rotation_y']
        label_files_candidate = []  # the scenes expected to be loaded.
        preprocess_data = []  # list of sub scene data in candidate scenes.
        for label_file in label_files:
            scene_num = int(str.split(label_file, sep='.')[0])
            if scene_num not in self.ignore_list:
                label_files_candidate.append(label_file)
        valid_length_sum = 0
        for label_file in label_files_candidate:
            file_path = os.path.join(label_files_path, label_file)
            raw_data = pd.read_table(file_path, sep=' ', header=None, names=names)
            raw_data_target = raw_data[raw_data['type'] == 'Pedestrian']
            # attribute calculate
            max_frame_id = raw_data['frame'].max()
            track_id_list = raw_data_target['track id'].unique().tolist()
            # initial store matrix
            data_matrix = np.zeros(
                (max_frame_id + 1, len(track_id_list), 1 + 4 + 3 + 3))  # bbox * 4 + dim * 3 + loc * 3

            # fill up a big matrix
            # 巨大随时间的长条矩阵，最终一个batch中的unit，本质上就是在长条上取一段seq_length出来。
            for _, row in raw_data_target.iterrows():
                index = track_id_list.index(row['track id'])
                frame = row['frame']
                data_matrix[frame, index, 0] = row['track id']
                data_matrix[frame, index, 1:5] = row['bbox_left'], row['bbox_top'], row['bbox_right'], row[
                    'bbox_bottom']
                data_matrix[frame, index, 5:8] = row['dimensions_x'], row['dimension_y'], row['dimension_z']
                data_matrix[frame, index, 8:11] = row['location_x'], row['location_y'], row['location_z']

            # take out useful time exists at least 1 person (in which the sub-matrix is not full of zero).
            # lambda expression: determine if sub-matrix x is a valid slice with valid object information.
            # 这一片时间片上是否有有效信息
            def is_useful(x):
                return np.sum(np.abs(x)) != 0

            # dump present slice into preprocess list.
            def dump(slice_buffer):
                if len(slice_buffer) != 0:
                    time_length = len(slice_buffer)
                    dump_dic = {'data': np.concatenate(slice_buffer, axis=0),
                                'file': label_file,
                                'start_frame': frame - time_length, 'end_length': frame - 1,
                                'id_by_index': track_id_list}
                    preprocess_data.append(dump_dic)

            data_matrix_slice = np.split(data_matrix, indices_or_sections=data_matrix.shape[0], axis=0)
            slice_buffer = []
            count = 0
            length_count = 0
            for (frame, slice_) in enumerate(data_matrix_slice):
                # This frame slice is useful (exist valid object)
                if is_useful(slice_):
                    slice_buffer.append(slice_)
                    # last step of the loop
                    if frame == len(data_matrix_slice) - 1 and len(slice_buffer) != 0:
                        count += 1
                        length_count += len(slice_buffer)
                        dump(slice_buffer)
                else:
                    # dump data in buffer and clean
                    if len(slice_buffer) != 0:
                        count += 1
                        length_count += len(slice_buffer)
                        dump(slice_buffer)
                    slice_buffer = []

            valid_length_sum += length_count
            print('{} sub scenes in {}, valid length = {}'.format(count, label_file, length_count))

        print('Total Valid Length = {}'.format(valid_length_sum))
        return preprocess_data

    def __len__(self):
        return self.num_sequence // self.batch_size

    def next_batch(self):
        batch_data_full = []
        batch_data_appendix = []
        for batch_i in range(0, self.batch_size):
            sub_scene = self.list_sub_scene_data[self.sub_scene_ptr]
            sub_scene_data = sub_scene['data']
            # check present sub scene availability. If not, goto next sub scene.
            if sub_scene_data.shape[0] < self.sequence_ptr + self.seq_length + 1:
                self.sub_scene_ptr += 1
                self.sequence_ptr = 0
                sub_scene = self.list_sub_scene_data[self.sub_scene_ptr]
                sub_scene_data = sub_scene['data']

            # form batch slice
            batch_slice = np.zeros((self.seq_length + 1, self.max_num_peds, 11))
            begin, end = self.sequence_ptr, self.sequence_ptr + self.seq_length + 1

            for i in range(0, len(sub_scene['id_by_index'])):
                batch_slice[:, i, :] = sub_scene_data[begin:end, i, :]

            # Add a batch slice to batch list
            batch_data_full.append(batch_slice)
            # Add appendix related to this batch slice, including scene_name, start_frame, end_frame, id_by_index
            appendix = sub_scene.copy()
            batch_data_appendix.append(appendix.pop('data'))
            # self add for next call
            self.sequence_ptr += 1

        # concatenate and split it into input and output
        batch_data_full = np.stack(batch_data_full, axis=0)
        batch_x = batch_data_full[:, 0:self.seq_length, :]
        batch_y = batch_data_full[:, 1:self.seq_length + 1, :]
        return batch_x, batch_y, batch_data_appendix

    def reset_ptr(self):
        self.sub_scene_ptr = 0
        self.sequence_ptr = 0


# 随机打印这个batch_slice中一个有有效轨迹行人的BB框，验证数据的正确性
def print_for_valid(batch_slice):
    # plt.plot([1, 2, 3], [1, 2, 3])
    plt.axis([0, 1512, 0, 600])
    for ped in range(0, batch_slice.shape[1]):
        ped_traj = np.squeeze(batch_slice[:, ped, :])  # seq_length, 11
        if np.sum(ped_traj) != 0:
            print(ped_traj)
            for frame in range(ped_traj.shape[0]):
                x, y = ped_traj[frame, 1], ped_traj[frame, 2]
                width, height = ped_traj[frame, 3] - ped_traj[frame, 1], ped_traj[frame, 4] - ped_traj[frame, 2]
                rect = plt.Rectangle((x, y), width, height, fill=False)
                plt.gca().add_patch(rect)
            plt.show()
            break


if __name__ == "__main__":
    kitti_loader = KittiDataLoader(16, 5, 70, [], sub_set='raw')
    batch_slice_choose = 6
    for i in range(0, len(kitti_loader)):
        x, y, _ = kitti_loader.next_batch()
        print_for_valid(x[batch_slice_choose])
        exit(0)
