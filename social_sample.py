import numpy as np
import tensorflow as tf

import os
import pickle
import argparse

from social_utils import SocialDataLoader
from social_model import SocialModel
from grid import getSequenceGridMask

import path_static


# from social_train import getSocialGrid, getSocialTensor


def get_mean_error(predicted_traj, true_traj, observed_length, maxNumPeds):
    '''
    Function that computes the mean euclidean distance error between the
    predicted and the true trajectory
    params:
    predicted_traj : numpy matrix with the points of the predicted trajectory
    true_traj : numpy matrix with the points of the true trajectory
    observed_length : The length of trajectory observed
    '''
    # The data structure to store all errors
    error = np.zeros(len(true_traj) - observed_length)
    # For each point in the predicted part of the trajectory
    for i in range(observed_length, len(true_traj)):
        # The predicted position. This will be a maxNumPeds x 3 matrix
        pred_pos = predicted_traj[i, :]
        # The true position. This will be a maxNumPeds x 3 matrix
        true_pos = true_traj[i, :]
        timestep_error = 0
        counter = 0
        for j in range(maxNumPeds):
            if true_pos[j, 0] == 0:
                # Non-existent ped
                continue
            else:
                timestep_error += np.linalg.norm(true_pos[j, [1, 2]] - pred_pos[j, [1, 2]])
                counter += 1

        error[i - observed_length] = timestep_error / counter

        # The euclidean distance is the error
        # error[i-observed_length] = np.linalg.norm(true_pos - pred_pos)

    # Return the mean error
    return np.mean(error)


def get_final_error(predicted_traj, true_traj, observed_length, maxNumPeds):
    '''
    Function that computes the final euclidean distance error between the
    predicted and the true trajectory
    params:
    predicted_traj : numpy matrix with the points of the predicted trajectory
    true_traj : numpy matrix with the points of the true trajectory
    observed_length : The length of trajectory observed
    '''
    # The data structure to store all errors
    error = np.zeros(1)
    # For each point in the predicted part of the trajectory
    for i in range(len(true_traj) - 1, len(true_traj)):
        # The predicted position. This will be a maxNumPeds x 3 matrix
        pred_pos = predicted_traj[i, :]
        # The true position. This will be a maxNumPeds x 3 matrix
        true_pos = true_traj[i, :]
        timestep_error = 0
        counter = 0
        for j in range(maxNumPeds):
            if true_pos[j, 0] == 0:
                # Non-existent ped
                continue
            else:
                timestep_error += np.linalg.norm(true_pos[j, [1, 2]] - pred_pos[j, [1, 2]])
                counter += 1

        error[0] = timestep_error / counter

        # The euclidean distance is the error
        # error[i-observed_length] = np.linalg.norm(true_pos - pred_pos)

    # Return the mean error
    return np.mean(error)


def trajectory_dict_generate(dataset, real_traj, gen_traj, frame_seq, obs_len):
    dict = {}
    dict['dataset'] = dataset
    dict['real_traj'] = real_traj
    dict['generate_traj'] = gen_traj
    dict['frame_seq'] = frame_seq
    dict['obs_length'] = obs_len
    return dict


def trajectory_record(dataset, real_traj_batch, gen_traj_batch, frame_batch, obs_length):
    maxNumPeds = real_traj_batch.shape[1]
    list = []
    for i in range(maxNumPeds):
        real_traj = np.squeeze(real_traj_batch[:, i, :])
        gen_traj = np.squeeze(gen_traj_batch[:, i, :])
        traj_dict_raw = trajectory_dict_generate(dataset, real_traj, gen_traj, frame_batch, obs_length)
        list.append(traj_dict_raw)

    return list


def evaluate(dataset, model, sess, sample_args, saved_args):
    # Create a SocialDataLoader object with batch_size 1 and seq_length equal to observed_length + pred_length
    data_loader = SocialDataLoader(1, sample_args.pred_length + sample_args.obs_length, saved_args.maxNumPeds, dataset,
                                   True)

    # Reset all pointers of the data_loader
    data_loader.reset_batch_pointer()

    # yzn : list for saving each trajectory information
    traj_list = []

    # Variable to maintain total error
    total_mean_error = 0
    total_final_error = 0
    # For each batch
    for b in range(data_loader.num_batches):
        # Get the source, target and dataset data for the next batch
        x, y, f, d = data_loader.next_batch()

        # Batch size is 1
        x_batch, y_batch, frame_batch, d_batch = x[0], y[0], f[0], d[0]

        if d_batch == 2 and dataset[0] == 2:
            print('Low scale scene found.')
            dimensions = [640, 480]
        else:
            dimensions = [1920, 1080]

        grid_batch = getSequenceGridMask(x_batch, dimensions, saved_args.neighborhood_size, saved_args.grid_size)

        obs_traj = x_batch[:sample_args.obs_length]
        obs_grid = grid_batch[:sample_args.obs_length]
        # obs_traj is an array of shape obs_length x maxNumPeds x 5

        complete_traj = model.sample(sess, obs_traj, obs_grid, dimensions, x_batch, sample_args.pred_length)

        # ipdb.set_trace()
        # complete_traj is an array of shape (obs_length+pred_length) x maxNumPeds x 3
        mean_error = get_mean_error(complete_traj, x[0], sample_args.obs_length, saved_args.maxNumPeds)
        final_error = get_final_error(complete_traj, x[0], sample_args.obs_length, saved_args.maxNumPeds)
        total_mean_error += mean_error
        total_final_error += final_error

        # trajectory dictionary generator
        list = trajectory_record(data_loader.get_mot16_subname(d_batch), x_batch, complete_traj, frame_batch,
                                 sample_args.obs_length)
        traj_list += list

        if b % 10 == 0:
            print("Processed trajectory number : ", b, "out of ", data_loader.num_batches, " trajectories")

    # Print the mean error across all the batches
    print("Total mean error of the model is ", total_mean_error / data_loader.num_batches)
    print("Total final error of the model is ", total_final_error / data_loader.num_batches)

    return traj_list


def main(test_dataset):
    parser = argparse.ArgumentParser()
    # Observed length of the trajectory parameter
    parser.add_argument('--obs_length', type=int, default=10,
                        help='Observed length of the trajectory')
    # Predicted length of the trajectory parameter
    parser.add_argument('--pred_length', type=int, default=10,
                        help='Predicted length of the trajectory')
    # Test dataset
    parser.add_argument('--test_dataset', type=int, default=0,
                        help='Dataset to be tested on')

    # Parse the parameters
    sample_args = parser.parse_args()

    savepath = path_static.save_path
    # Define the path for the config file for saved args
    with open(os.path.join(savepath, 'social_config.pkl'), 'rb') as f:
        saved_args = pickle.load(f)
    # Create a SocialModel object with the saved_args and infer set to true
    model = SocialModel(saved_args, True)
    # Initialize a TensorFlow session
    sess = tf.InteractiveSession()
    # Initialize a saver
    saver = tf.train.Saver()

    # Get the checkpoint state for the model
    ckpt = tf.train.get_checkpoint_state(savepath)
    print('loading model: ', ckpt.model_checkpoint_path)

    # Restore the model at the checkpoint
    saver.restore(sess, ckpt.model_checkpoint_path)

    # 用于记录测试数据库的轨迹数据
    traj_list = []

    # Dataset to get data from
    # dataset = [sample_args.test_dataset]
    dataset = test_dataset
    for i in dataset:
        list = evaluate([i], model, sess, sample_args, saved_args)
        traj_list += list

    # save trajectory
    traj_file = open(os.path.join(savepath, 'traj_file_raw'), 'wb')
    pickle.dump(traj_list, traj_file)


if __name__ == '__main__':
    # test_datasets = [4]
    test_datasets = range(0, 6)
    main(test_datasets)
