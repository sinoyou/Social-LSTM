import os
import pickle
import string

import numpy as np
import pandas as pd


class KittiDataLoader():
    def __init__(self, batch_size, seq_length, max_num_peds, scenes_list, database_dir='data\\KITTI'):
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
        :param scenes_list: number of scenes in Kitti Database.
        """
        # save parameters
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.max_num_peds = max_num_peds
        self.scenes_list = scenes_list
        self.database_dir = database_dir

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

    def preprocess(self):
        """
        pre process *.txt data to list of maxtirx
        :return: list of matrix
        """
        # load data -> preprocess_files_candidate
        label_files_path = os.path.join(self.database_dir, 'training', 'label_02')
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
            if scene_num in self.scenes_list:
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
            print('{} sub scenes in {}, valid length = {}'.format(count, label_file, length_count))

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
            batch_slice[:, 0:len(sub_scene['id_by_index']), :] = sub_scene_data[begin:end, :, :]
            # Add a batch slice to batch list
            batch_data_full.append(batch_slice)
            # Add appendix related to this batch slice, including scene_name, start_frame, end_frame, id_by_index
            appendix = sub_scene.copy()
            batch_data_appendix.append(appendix.pop('data'))
            # self add for next call
            self.sequence_ptr += 1

        # concatenate and split it into input and output
        batch_data_full = np.concatenate(batch_data_full, axis=0)
        batch_x = batch_data_full[0:self.seq_length, :]
        batch_y = batch_data_full[1:self.seq_length + 1, :]
        return batch_x, batch_y, batch_data_appendix

    def reset_ptr(self):
        self.sub_scene_ptr = 0
        self.sequence_ptr = 0


if __name__ == "__main__":
    kitti_loader = KittiDataLoader(32, 5, 70, range(0, 21))
    for i in range(0, len(kitti_loader)):
        x, y, _ = kitti_loader.next_batch()
