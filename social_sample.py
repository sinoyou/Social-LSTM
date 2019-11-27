import numpy as np
import tensorflow as tf

import os
import pickle
import argparse
import json

from deprecated.social_utils_mot16 import SocialDataLoader
from social_model import SocialModel
from grid import getSequenceGridMask

import path_static

# from social_train import getSocialGrid, getSocialTensor
from social_utils_kitti import KittiDataLoader, data_filter_location_and_bb


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


def evaluate(model, sess, sample_args, saved_args):
    # Create a SocialDataLoader object with batch_size 1 and seq_length equal to observed_length + pred_length
    data_loader = KittiDataLoader(1, sample_args.obs_length + sample_args.pred_length, saved_args.maxNumPeds, [],
                                  sub_set='test')

    # Reset all pointers of the data_loader
    data_loader.reset_ptr()

    # yzn : list for saving each trajectory information
    traj_list = []

    # Variable to maintain total error
    total_mean_error = 0
    total_final_error = 0

    json_result = []

    # For each batch
    for b in range(len(data_loader)):
        # Get the source, target and dataset data for the next batch
        x, y, appendix = data_loader.next_batch()

        # Batch size is 1
        x_batch, y_batch = x[0], y[0]

        x_batch, x_rel_batch = data_filter_location_and_bb(x_batch)
        y_batch, y_rel_batch = data_filter_location_and_bb(y_batch)

        grid_batch = getSequenceGridMask(x_batch, saved_args.neighborhood_size, saved_args.grid_size)

        # relative process
        if saved_args.relative_path:
            input_x = x_rel_batch
            input_y = y_rel_batch
        else:
            input_x = x_batch
            input_y = y_batch

        obs_traj = input_x[:sample_args.obs_length]
        obs_grid = grid_batch[:sample_args.obs_length]
        # obs_traj is an array of shape obs_length x maxNumPeds x 5

        complete_traj = model.sample(sess, obs_traj, obs_grid, input_x, sample_args.pred_length)

        def rel_to_abs(x):
            result = np.zeros_like(x)
            for i in range(1, result.shape[0]):
                result[i, :] = result[i - 1, :] + x[i, :]
            return result

        if saved_args.relative_path:
            cmp_pred = rel_to_abs(complete_traj)
            cmp_true = rel_to_abs(input_x)
        else:
            cmp_pred = complete_traj
            cmp_true = input_x

        # ipdb.set_trace()
        # complete_traj is an array of shape (obs_length+pred_length) x maxNumPeds x 3

        json_result.append((cmp_true, cmp_pred, appendix[0]))

        mean_error = get_mean_error(cmp_pred, cmp_true, sample_args.obs_length,
                                    saved_args.maxNumPeds)
        final_error = get_final_error(cmp_pred, cmp_true, sample_args.obs_length,
                                      saved_args.maxNumPeds)
        total_mean_error += mean_error
        total_final_error += final_error

        if b % 10 == 0:
            print("Processed trajectory number : ", b, "out of ", len(data_loader), " trajectories")

    # Print the mean error across all the batches
    print("Total mean error of the model is ", total_mean_error / len(data_loader))
    print("Total final error of the model is ", total_final_error / len(data_loader))

    return json_result


def main():
    parser = argparse.ArgumentParser()
    # Observed length of the trajectory parameter
    parser.add_argument('--obs_length', type=int, default=5,
                        help='Observed length of the trajectory')
    # Predicted length of the trajectory parameter
    parser.add_argument('--pred_length', type=int, default=3,
                        help='Predicted length of the trajectory')
    # Save path
    parser.add_argument('--save_path', type=str, help='Path of saving trained model.')

    # Parse the parameters
    sample_args = parser.parse_args()

    savepath = sample_args.save_path
    # Define the path for the config file for saved args
    with open(os.path.join(savepath, 'social_config.pkl'), 'rb') as f:
        saved_args = pickle.load(f)
    # Create a SocialModel object with the saved_args and infer set to true
    model = SocialModel(saved_args, infer=True)
    # Initialize a TensorFlow session
    sess = tf.InteractiveSession()
    # Initialize a saver
    saver = tf.train.Saver()

    # Get the checkpoint state for the model
    ckpt = tf.train.get_checkpoint_state(savepath)
    print('loading model: ', ckpt.model_checkpoint_path)

    # Restore the model at the checkpoint
    saver.restore(sess, ckpt.model_checkpoint_path)

    # Dataset to get data from
    json_result = evaluate(model, sess, sample_args, saved_args)

    with open(os.path.join(sample_args.save_path, 'traj.json'), 'r') as f:
        f.write(json.dumps(json_result))


if __name__ == '__main__':
    main()
