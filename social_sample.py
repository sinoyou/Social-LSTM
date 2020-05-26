import numpy as np
import tensorflow as tf

import os
import pickle
import argparse
import json

from social_model import SocialModel
from grid import getSequenceGridMask

# from social_train import getSocialGrid, getSocialTensor
# from social_utils_kitti import KittiDataLoader
from data.csv_dataloader import SocialDataLoader
from tools import Recorder


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
    error_list = list()

    for p in range(maxNumPeds):
        p_pred_traj = predicted_traj[observed_length:, p, :]
        p_true_traj = true_traj[observed_length:, p, :]

        p_pred_traj_valid = p_pred_traj[(p_pred_traj[:, 0] > 0) & (p_true_traj[:, 0] > 0), 1:3]
        p_true_traj_valid = p_true_traj[(p_pred_traj[:, 0] > 0) & (p_true_traj[:, 0] > 0), 1:3]

        p_error_vector = np.sqrt(np.sum((p_pred_traj_valid - p_true_traj_valid) ** 2, axis=-1))

        # if exist valid parts
        if p_error_vector.shape[0] > 0:
            error_list.append(np.mean(p_error_vector))

    # Return the mean error
    return error_list


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
    error_list = list()

    for p in range(maxNumPeds):
        p_pred_traj = predicted_traj[observed_length:, p, :]
        p_true_traj = true_traj[observed_length:, p, :]

        p_pred_traj_valid = p_pred_traj[(p_pred_traj[:, 0] > 0) & (p_true_traj[:, 0] > 0), 1:3]
        p_true_traj_valid = p_true_traj[(p_pred_traj[:, 0] > 0) & (p_true_traj[:, 0] > 0), 1:3]

        p_error_vector = np.sqrt(np.sum((p_pred_traj_valid - p_true_traj_valid) ** 2, axis=-1))
        # if exist valid parts
        if p_error_vector.shape[0] > 0:
            # only append in 'reach to the tail' mode.
            if p_pred_traj[-1, 0] > 0 and p_true_traj[-1, 0] > 0:
                error_list.append(p_error_vector[-1])

    # Return the mean error
    return error_list


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


def evaluate(model, sess, sample_args, saved_args, recorder):
    # Create a SocialDataLoader object with batch_size 1 and seq_length equal to observed_length + pred_length
    # data_loader = KittiDataLoader(1, sample_args.obs_length + sample_args.pred_length, saved_args.maxNumPeds, [],
    #                               sub_set='test')
    data_loader = SocialDataLoader(file_path=sample_args.test_dataset,
                                   batch_size=1,
                                   seq_length=sample_args.obs_length + sample_args.pred_length,
                                   max_num_peds=saved_args.maxNumPeds,
                                   mode='valid',
                                   train_leave=saved_args.train_leave,
                                   valid_scene=sample_args.valid_scene,
                                   recorder=recorder)

    # Reset all pointers of the data_loader
    data_loader.reset_ptr()

    # yzn : list for saving each trajectory information
    traj_list = []

    # Variable to maintain total error
    total_mean_error = list()
    total_final_error = list()

    # For each batch
    for b in range(len(data_loader)):
        # Get the source, target and dataset data for the next batch
        data = data_loader.next_batch()
        x, y = data[:, :sample_args.obs_length], data[:, sample_args.obs_length:]

        # Batch size is 1
        x_batch, y_batch, xy_batch = x[0], y[0], data[0]

        grid_batch = getSequenceGridMask(x_batch, saved_args.neighborhood_size, saved_args.grid_size)

        # relative process
        if saved_args.relative_path:
            raise Exception('Relative not implemented.')
        else:
            input_x = x_batch
            input_y = y_batch

        obs_traj = input_x
        obs_grid = grid_batch

        # obs_traj is an array of shape obs_length x maxNumPeds x 3
        complete_traj = model.sample(sess, obs_traj, obs_grid, xy_batch, sample_args.pred_length,
                                     data_loader=data_loader)

        if saved_args.relative_path:
            raise Exception('Relative not implemented.')
        else:
            # complete_traj is an array of shape (obs_length+pred_length) x maxNumPeds x 3
            cmp_pred = complete_traj
            cmp_true = xy_batch

        raw_cmp_pred = data_loader.norm_to_raw(cmp_pred)
        raw_cmp_true = data_loader.norm_to_raw(cmp_true)
        mean_error = get_mean_error(raw_cmp_pred, raw_cmp_true, sample_args.obs_length,
                                    saved_args.maxNumPeds)
        final_error = get_final_error(raw_cmp_pred, raw_cmp_true, sample_args.obs_length,
                                      saved_args.maxNumPeds)

        total_mean_error += mean_error
        total_final_error += final_error

        if b % 10 == 0:
            recorder.logger.info("Processed trajectory number : {} out of {} trajectories".format(b, len(data_loader)))

    # Print the mean error across all the batches
    recorder.logger.info("Total mean error of the model is {}".format(sum(total_mean_error) / len(total_mean_error)))
    recorder.logger.info("Total final error of the model is {}".format(sum(total_final_error) / len(total_final_error)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_dataset', type=str, default='data/KITTI/kitti-all-label02.csv',
                        help='Path of test data sets.')
    parser.add_argument('--valid_scene', nargs='+', type=int,
                        help='scenes for validation')
    # Observed length of the trajectory parameter
    parser.add_argument('--obs_length', type=int, default=6,
                        help='Observed length of the trajectory')
    # Predicted length of the trajectory parameter
    parser.add_argument('--pred_length', type=int, default=8,
                        help='Predicted length of the trajectory')
    parser.add_argument('--phase', type=str, default='test')
    # Save path
    parser.add_argument('--save_path', type=str, default='save/',
                        help='Path of saving trained model.')
    # log
    parser.add_argument('--board_name', type=str, default='runs/test/')

    # Parse the parameters
    sample_args = parser.parse_args()

    # recorder
    recorder = Recorder(summary_path=sample_args.board_name, board=True)

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
    recorder.logger.info('loading model: ' + str(ckpt.model_checkpoint_path))

    # Restore the model at the checkpoint
    saver.restore(sess, ckpt.model_checkpoint_path)

    # Dataset to get data from
    evaluate(model, sess, sample_args, saved_args, recorder)

    # with open(os.path.join(sample_args.save_path, 'traj.json'), 'r') as f:
    #     f.write(json.dumps(json_result))


if __name__ == '__main__':
    main()
