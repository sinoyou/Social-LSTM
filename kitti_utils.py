import os
import pickle
import random
import string

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


class KittiDataLoader:
    def __init__(self, batch_size, seq_length, ignore_list, sub_set, database_dir='data\KITTI'):
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
        self.ignore_list = ignore_list
        self.database_dir = database_dir
        self.seq_ptr = 0
        self.track_ptr = 0
        self.sub_set = sub_set

        # pre process data
        list_preprocess_data = self.preprocess()

        # select valid processed sub-sequence of scene with seq_length larger than seq_length
        self.filter_data = [x for x in list_preprocess_data if
                            x.shape[0] >= self.seq_length + 1]
        random.shuffle(self.filter_data)

        # calculate data's attributes.
        self.num_sequence = 0
        for data in self.filter_data:
            self.num_sequence += data.shape[0] - (self.seq_length + 1) + 1

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

            # Vanilla LSTM's 数据生成
            if data_matrix.shape[1] == 0:
                continue
            tracks = np.split(data_matrix, data_matrix.shape[1], axis=1)
            for track in tracks:
                # may contain zero, so only filter the useful one.
                track = np.squeeze(track, axis=1)
                sum_ = np.sum(track, axis=1)
                valid_track = track[sum_ != 0, :]
                head = 0
                tail = track.shape[0] - 1
                while sum_[head] == 0 and head < track.shape[0] - 1:
                    head += 1
                while sum_[tail] == 0 and tail > 0:
                    tail -= 1
                if tail - head + 1 != valid_track.shape[0]:
                    print(track)
                    print('Track None Zero Length is {} != Continuous Length is {}'.
                          format(valid_track.shape[0], tail - head + 1))
                preprocess_data.append(valid_track)

        cnt = 0
        for track in preprocess_data:
            cnt += track.shape[0]

        print('Total Track Number {}'.format(cnt))
        return preprocess_data

    def __len__(self):
        return self.num_sequence // self.batch_size

    def next_batch(self):
        batch_data_full = []
        while len(batch_data_full) < self.batch_size:
            full_track = self.filter_data[self.track_ptr]
            if self.seq_length + self.seq_ptr >= full_track.shape[0]:
                self.seq_ptr = 0
                self.track_ptr += 1
            batch_data_full.append(full_track[self.seq_ptr:self.seq_ptr + self.seq_length + 1, :])

        # concatenate and split it into input and output
        batch_data_full = np.stack(batch_data_full, axis=0)
        batch_x = batch_data_full[:, 0:self.seq_length, :]
        batch_y = batch_data_full[:, 1:self.seq_length + 1, :]
        return batch_x, batch_y

    def reset_ptr(self):
        self.seq_ptr = 0
        self.track_ptr = 0


# 随机打印这个batch_slice中一个有有效轨迹行人的BB框，验证数据的正确性
def print_for_valid(seq):
    plt.subplot(2, 1, 1)
    plt.plot(seq[:, 8], seq[:, 10])
    plt.subplot(2, 1, 2)
    plt.axis([0, 1512, 0, 600])
    seq_slices = np.squeeze(np.split(seq, seq.shape[0], axis=0))
    for seq_slice in seq_slices:
        left_x, left_y = seq_slice[1], seq_slice[2]
        width = seq_slice[3] - seq_slice[1]
        height = seq_slice[4] - seq_slice[2]
        rec = plt.Rectangle((left_x, left_y), width, height, fill=False)
        plt.gca().add_patch(rec)
    plt.show()


if __name__ == "__main__":
    kitti_loader = KittiDataLoader(16, 10, [], sub_set='train')
    for i in range(0, len(kitti_loader)):
        x, y = kitti_loader.next_batch()
        print_for_valid(x[0])
        exit(0)
