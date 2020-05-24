import os
import numpy as np
import pandas as pd
import random
from tools import Recorder

class KittiSocialDataLoader():
    def __init__(self, file_path, batch_size, seq_length, max_num_peds, mode, train_leave, recorder, seed=17373321,
                 valid_scene=None, fragment=False):
        """
        KITTI dataloader for social LSTM. Support csv format file.
        :param file_path: csv file path
        :param seq_length: length of sequence (obs_length + pred_length), you need to devide returned data manually. 
        :param max_num_peds: maximum number of pedestrians.
        :train_leave: list(), which scenes are left not for training. 
        :param mode: 'train' or 'valid', to decide returned data.
        :param recorder: log writer. 
        :param seed: random seed.
        :param valid_scene: list(), which scenes are used for validation. 
        :param fragment: when the last left data not enough for a batch, if return it.
        """
        self.batch_size = batch_size
        self.seq_len = seq_length
        self.max_num_peds = max_num_peds
        self.fragment = fragment
        self.mode = mode
        self.recorder = recorder
        self.batch_ptr = 0

        # read raw data.
        raw_data = pd.read_csv(file_path)

        # args check
        assert not (self.mode == 'valid' and valid_scene is None)
        assert self.mode in ['train', 'valid']

        # get train data
        if train_leave is not None:
            if isinstance(train_leave, int):
                train_leave = [train_leave]
            self.recorder.logger.info('Scenes {} are left not for training.'.format(train_leave))
            leaves = [raw_data['scene'] == s for s in train_leave]
            mask = leaves[0]
            for i in range(1, len(leaves)):
                mask = mask | leaves[i]
            self.train_data = raw_data[~mask]
        else:
            self.train_data = raw_data

        # get valid data
        if valid_scene is not None:
            if isinstance(valid_scene, int):
                valid_scene = [valid_scene]
            # valid scene
            self.recorder.logger.info('Scenes {} are used for validation.'.format(valid_scene))
            targets = [raw_data['scene'] == s for s in valid_scene]
            mask = targets[0]
            for i in range(1, len(targets)):
                mask = mask | targets[i]
            self.valid_data = raw_data[mask]
        else:
            self.valid_data = None

        # get mean and std from training data.
        self.norm_targets = ['loc_x', 'loc_y', 'loc_z']
        self.norm_metric = dict()
        self.get_mean_std()

        if self.mode == 'train':
            self.data = self.preprocess(self.train_data)
        else:
            self.data = self.preprocess(self.valid_data)

        random.seed(seed)
        random.shuffle(self.data, random=random.random)

        # print summary.
        self.recorder.logger.info('Count = {}, Batch Size = {}, Iteration = {}'.format(
            len(self.data), self.batch_size, self.__len__()
        ))

        if self.__len__() <= 0:
            raise Exception('No enough data for a batch size = {}'.format(self.batch_size))

    def preprocess(self, filter_raw):
        """
        process DataFrame to a list of ([seq_len, max_num_peds, 3])
        """
        data = list()

        # norm process
        def norm(row):
            for target in self.norm_targets:
                row[target] = (row[target] - self.norm_metric[target + '_mean']) / self.norm_metric[target + '_std']
            return row

        filter_raw = filter_raw.apply(norm, axis=1)

        scenes = filter_raw['scene'].unique()
        for scene in scenes:
            scene_df = filter_raw[filter_raw['scene'] == scene]
            max_frame = scene_df['frame'].max()

            # define a Tensor to hold all vru's info in one scenes
            scene_tensor = np.zeros((max_frame + 1, self.max_num_peds, 3))

            # Hint: difference between index / vru / unique_id
            # vru: original object in in KITTI.
            # unique_id: '1{scene:2d}{vru_id:3d}', to uniquely identify a vru. Used as value in social Tensor.
            # index: range(0, len(vru_ids)). Used as index in social Tensor, as unique_id may be not continuous.
            vru_ids = scene_df['id'].unique()
            for index, vru in enumerate(vru_ids):
                vru_frames = scene_df[scene_df['id'] == vru]['frame']
                vru_traj = scene_df[scene_df['id'] == vru][['loc_x', 'loc_z']]
                # Due to scene and id in raw data can be zero, so a none zero id is needed. 
                unique_id = self.get_unique_id(scene, vru)
                scene_tensor[list(vru_frames), index, 0] = unique_id
                scene_tensor[list(vru_frames), index, 1:3] = vru_traj

            # slice scene_tensor to [seq_len, max_num_peds, 3]
            # Notice: Unlike UCY/ETH, due to disconnection of frames containing vru,
            # social_tensor[frame, :, :] may be zeros.
            ptr = 0
            while ptr + self.seq_len <= scene_tensor.shape[0]:
                # if all zero at first seq, then skip
                if np.sum(scene_tensor[ptr, :]) == 0:
                    ptr += 1
                # contains valid vru info, genearate a sequence
                else:
                    unit = scene_tensor[ptr:ptr + self.seq_len, :]
                    data.append(unit)
                    # todo ptr += 1 or ptr += seq_len
                    ptr += self.seq_len

        return data

    def __len__(self):
        if self.fragment:
            if len(self.data) % self.batch_size == 0:
                return len(self.data) // self.batch_size
            else:
                return len(self.data) // self.batch_size + 1
        else:
            return len(self.data) // self.batch_size

    def next_batch(self):
        if self.batch_ptr + self.batch_size <= len(self.data):
            batch_data = np.stack(self.data[self.batch_ptr:self.batch_ptr + self.batch_size], axis=0)
            self.batch_ptr += self.batch_size
        else:
            if self.fragment:
                batch_data = np.stack(self.data[self.batch_ptr:], axis=0)
            else:
                raise Exception('No Complete Batch Data index = {} + batch_size = {} > count = {}'.format(
                    self.batch_ptr, self.batch_size, len(self.data)
                ))
        return batch_data

    def reset_ptr(self):
        self.batch_ptr = 0

    def get_mean_std(self):
        for target in self.norm_targets:
            self.norm_metric[target + '_mean'] = self.train_data[target].mean()
            self.norm_metric[target + '_std'] = self.train_data[target].std()
        self.recorder.logger.info('Norm Metric {}'.format(self.norm_metric))

    def get_unique_id(self, scene, true_ids):
        return 1e5 + scene * 1e4 + true_ids

    def get_true_id(self, unique_ids):
        return unique_ids % 1e3


if __name__ == '__main__':
    file_path = 'kitti-all-label02.csv'
    recorder = Recorder(summary_path='runs', board=False, logfile=False, stream=True)
    loader = KittiSocialDataLoader(file_path=file_path, batch_size=5, seq_length=12, max_num_peds=80, mode='train',
                                   train_leave=None, recorder=recorder, valid_scene=None, fragment=True)
    x = loader.next_batch()
    print(x[0])
