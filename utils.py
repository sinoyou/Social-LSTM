import os
import pickle
import numpy as np
import random


# TODO : (improve) Add functionality to retrieve data only from specific datasets
class DataLoader():

    def __init__(self, batch_size=50, seq_length=5, datasets=[0, 1, 2, 3, 4], forcePreProcess=False):
        '''
        Initialiser function for the DataLoader class
        params:
        batch_size : Size of the mini-batch
        seq_length : RNN sequence length
        '''
        # List of data directories where raw data resides
        # self.data_dirs = ['./data/eth/univ', './data/eth/hotel',
        #                  './data/ucy/zara/zara01', './data/ucy/zara/zara02',
        #                  './data/ucy/univ']
        self.data_dirs = ['../data/eth/univ', '../data/eth/hotel']

        self.used_data_dirs = [self.data_dirs[x] for x in datasets]

        # Data directory where the pre-processed pickle file resides
        # yzn pickle：
        self.data_dir = '../data'

        # Store the batch size and the sequence length arguments
        self.batch_size = batch_size
        self.seq_length = seq_length

        # Define the path of the file in which the data needs to be stored
        data_file = os.path.join(self.data_dir, "trajectories.cpkl")

        # If the file doesn't exist already or if forcePreProcess is true
        if not (os.path.exists(data_file)) or forcePreProcess:
            print("Creating pre-processed data from raw data")
            # Preprocess the data from the csv files
            self.preprocess(self.used_data_dirs, data_file)

        # Load the data from the pickled file
        self.load_preprocessed(data_file)
        # Reset all the pointers
        self.reset_batch_pointer()

    def preprocess(self, data_dirs, data_file):
        '''
        The function that pre-processes the pixel_pos.csv files of each dataset
        into data that can be used
        params:
        data_dirs : List of directories where raw data resides
        data_file : The file into which all the pre-processed data needs to be stored
        '''
        # all_ped_data would be a dictionary with mapping from each ped to their
        # trajectories given by matrix 3 x numPoints with each column
        # in the order x, y, frameId
        # Pedestrians from all datasets are combined
        # Dataset pedestrian indices are stored in dataset_indices
        # yzn all_ped_data : {行人编号 : [x, y, 时间页框号]}
        all_ped_data = {}
        # yzn dataset_indices : 记录不同的数据目录（dir）加载出来的行人数量，以区分all_ped_data中的信息。
        dataset_indices = []
        current_ped = 0
        # For each dataset
        for directory in data_dirs:
            # Define the path to its respective csv file
            file_path = os.path.join(directory, 'pixel_pos.csv')

            # Load data from the csv file
            # Data is a 4 x numTrajPoints matrix
            # where each column is a (frameId, pedId, y, x) vector
            data = np.genfromtxt(file_path, delimiter=',')

            # Get the number of pedestrians in the current dataset
            numPeds = np.size(np.unique(data[1, :]))

            # For each pedestrian in the dataset
            for ped in range(1, numPeds + 1):
                # Extract trajectory of the current ped
                traj = data[:, data[1, :] == ped]
                # Format it as (x, y, frameId)
                traj = traj[[3, 2, 0], :]

                # Store this in the dictionary
                all_ped_data[current_ped + ped] = traj

            # Current dataset done
            dataset_indices.append(current_ped + numPeds)
            current_ped += numPeds

        # The complete data is a tuple of all pedestrian data, and dataset ped indices
        complete_data = (all_ped_data, dataset_indices)
        # Store the complete data into the pickle file
        f = open(data_file, "wb")
        pickle.dump(complete_data, f, protocol=2)
        f.close()

    def load_preprocessed(self, data_file):
        '''
        Function to load the pre-processed data into the DataLoader object
        params:
        data_file : The path to the pickled data file
        '''
        # Load data from the pickled file
        f = open(data_file, "rb")
        self.raw_data = pickle.load(f)
        f.close()

        # Get the pedestrian data from the pickle file
        all_ped_data = self.raw_data[0]
        # Not using dataset_indices for now
        # dataset_indices = self.raw_data[1]

        # Construct the data with sequences(or trajectories) longer than seq_length
        self.data = []
        counter = 0

        # For each pedestrian in the data
        # yzn 此处从预处理csv后生成的数据中取出了 轨迹长度大于rnn模型长度+2的行人 作为能够使用的数据 -> 放入data中
        for ped in all_ped_data:
            # Extract his trajectory
            traj = all_ped_data[ped]
            # If the length of the trajectory is greater than seq_length (+2 as we need both source and target data)
            if traj.shape[1] > (self.seq_length + 2):
                # TODO: (Improve) Store only the (x,y) coordinates for now
                # yzn : [空间信息，时间信息] -> [时间信息，空间信息]
                self.data.append(traj[[0, 1], :].T)
                # Number of batches this datapoint is worth
                counter += int(traj.shape[1] / ((self.seq_length + 2)))

        # Calculate the number of batches (each of batch_size) in the data
        self.num_batches = int(counter / self.batch_size)

    def next_batch(self):
        '''
        Function to get the next batch of points
        '''
        # yzn : 在data中的行人轨迹信息都是长度大于seq_length+2的，在生成x_batch和y_batch时用随机因子的方式选取轨迹信息中
        # 长度为seq_length的段作为最终的数据。
        # List of source and target data for the current batch
        x_batch = []
        y_batch = []
        # For each sequence in the batch
        for i in range(self.batch_size):
            # Extract the trajectory of the pedestrian pointed out by self.pointer
            traj = self.data[self.pointer]
            # Number of sequences corresponding to his trajectory
            n_batch = int(traj.shape[0] / (self.seq_length + 2))
            # Randomly sample a index from which his trajectory is to be considered
            idx = random.randint(0, traj.shape[0] - self.seq_length - 2)
            # Append the trajectory from idx until seq_length into source and target data
            x_batch.append(np.copy(traj[idx:idx + self.seq_length, :]))
            y_batch.append(np.copy(traj[idx + 1:idx + self.seq_length + 1, :]))

            # yzn 当一个行人的轨迹很长时，那么其被多次采样在一个batch中的概率也随之增大（注意只是概率增大了）
            if random.random() < (1.0 / float(n_batch)):
                # Adjust sampling probability
                # if this is a long datapoint, sample this data more with
                # higher probability
                self.tick_batch_pointer()

        return x_batch, y_batch

    def tick_batch_pointer(self):
        '''
        Advance the data pointer
        '''
        self.pointer += 1
        if (self.pointer >= len(self.data)):
            self.pointer = 0

    def reset_batch_pointer(self):
        '''
        Reset the data pointer
        '''
        self.pointer = 0


if __name__ == '__main__':
    dataLoader = DataLoader(datasets=[1], forcePreProcess=True)
