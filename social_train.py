import tensorflow as tf
import argparse
import os
import time
import pickle
import numpy as np

from social_model import SocialModel
from grid import getSequenceGridMask
# from social_utils_kitti import KittiDataLoader
from data.csv_dataloader import SocialDataLoader
from tools import Recorder


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dataset', type=str, default='data/KITTI/kitti-all-label02.csv',
                        help='path of dataset with csv format')
    parser.add_argument('--train_leave', default=None, nargs='+', type=int,
                        help='scenes are left not for training.')
    # log
    parser.add_argument('--phase', default='train', type=str,
                        help='header of board scalar.')
    parser.add_argument('--board_name', default='runs/', type=str,
                        help='path of saving summary and log.')
    # RNN size parameter (dimension of the output/hidden state)
    parser.add_argument('--rnn_size', type=int, default=128,
                        help='size of RNN hidden state')
    # TODO: (improve) Number of layers not used. Only a single layer implemented
    # Number of layers parameter
    parser.add_argument('--num_layers', type=int, default=1,
                        help='number of layers in the RNN')
    # Model currently not used. Only LSTM implemented
    # Type of recurrent unit parameter
    parser.add_argument('--model', type=str, default='lstm',
                        help='rnn, gru, or lstm')
    # Size of each batch parameter
    parser.add_argument('--batch_size', type=int, default=3,
                        help='minibatch size')
    # Length of sequence to be considered parameter
    parser.add_argument('--seq_length', type=int, default=12,
                        help='RNN sequence length')
    # Number of epochs parameter
    parser.add_argument('--num_epochs', type=int, default=101,
                        help='number of epochs')
    # Frequency at which the model should be saved parameter
    parser.add_argument('--save_every', type=int, default=10,
                        help='save frequency')
    # TODO: (resolve) Clipping gradients for now. No idea whether we should
    # Gradient value at which it should be clipped
    parser.add_argument('--grad_clip', type=float, default=1.5,
                        help='clip gradients at this value')
    # Learning rate parameter
    parser.add_argument('--learning_rate', type=float, default=3e-3,
                        help='learning rate')
    # Decay rate for the learning rate parameter
    parser.add_argument('--decay_rate', type=float, default=5e-5,
                        help='decay rate for rmsprop')
    # Dropout not implemented.
    # Dropout probability parameter
    parser.add_argument('--keep_prob', type=float, default=1.0,
                        help='dropout keep probability')
    # Dimension of the embeddings parameter
    parser.add_argument('--embedding_size', type=int, default=64,
                        help='Embedding dimension for the spatial coordinates')
    # Size of neighborhood to be considered parameter
    parser.add_argument('--neighborhood_size', type=int, default=32,
                        help='Neighborhood size to be considered for social grid')
    # Size of the social grid parameter
    parser.add_argument('--grid_size', type=int, default=8,
                        help='Grid size of the social grid')
    # Maximum number of pedestrians to be considered
    parser.add_argument('--maxNumPeds', type=int, default=20,
                        help='Maximum Number of Pedestrians')
    # Save place
    parser.add_argument('--save_dir', type=str, default='save/',
                        help='directory of saving ckpt, log and config.')
    # Visible Device
    parser.add_argument('--device', type=str, default='0', help='GPU device num')
    # Use relative path
    parser.add_argument('--relative_path', type=bool, default=False,
                        help='Use relative path as obs and pred, default True')
    args = parser.parse_args()

    recorder = Recorder(summary_path=args.board_name)

    recorder.logger.info(args)
    train(args, recorder=recorder)


def train(args, recorder):
    # set visible cuda device
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device

    # Create the SocialDataLoader object
    # data_loader = KittiDataLoader(args.batch_size, args.seq_length, args.maxNumPeds, ignore_list=[], sub_set='train')
    data_loader = SocialDataLoader(file_path=args.train_dataset,
                                   batch_size=args.batch_size,
                                   seq_length=args.seq_length + 1,
                                   max_num_peds=args.maxNumPeds,
                                   mode='train',
                                   train_leave=args.train_leave,
                                   recorder=recorder)

    savepath = args.save_dir
    # save path check 当保存目录已经存在时需要特别处理，以防止保存的模型出现覆盖
    if os.path.exists(savepath):
        print("[WARNING]: Save Path Already Exists. Do you want to continue ? ")
        # command = input('[y/n]:')
        command = 'y'
        if len(command) == 1 and command.lower()[0] == 'y':
            pass
        else:
            exit(0)
    else:
        os.mkdir(savepath)

    # Initialize a TensorFlow session
    with tf.Session() as sess:

        # 模型初始化或预加载
        def get_model(force):
            if not force and os.path.exists(os.path.join(savepath, 'social_config.pkl')):
                with open(os.path.join(savepath, 'social_config.pkl'), 'rb') as f:
                    save_args = pickle.load(f)
                model = SocialModel(save_args)
                # Restore variables from checkpoint
                ckpt = tf.train.get_checkpoint_state(savepath)
                recorder.logger.info('loading model: ', ckpt.model_checkpoint_path)
                saver = tf.train.Saver()
                saver.restore(sess, save_path=ckpt.model_checkpoint_path)
            else:
                with open(os.path.join(savepath, 'social_config.pkl'), 'wb') as f:
                    pickle.dump(args, f)
                # Create a SocialModel object with the arguments
                model = SocialModel(args)
                # Initialize all variables in the graph
                sess.run(tf.global_variables_initializer())
                # Initialize a saver that saves all the variables in the graph
                saver = tf.train.Saver(tf.global_variables())

            return model, saver

        model, saver = get_model(force=True)

        # For each epoch
        for e in range(args.num_epochs):
            # Assign the learning rate value for this epoch
            sess.run(tf.assign(model.lr, args.learning_rate * (args.decay_rate ** e)))
            # Reset the data pointers in the data_loader
            data_loader.reset_ptr()

            # For each batch
            for b in range(len(data_loader)):
                # Tic
                start = time.time()

                # Get the source, target and appendix data for the next batch
                data = data_loader.next_batch()
                x, y = data[:, :-1], data[:, 1:]

                # variable to store the loss for this batch
                loss_batch = 0

                # variables to store information of loss counter
                counter_batch = 0
                full_counter_batch = 0

                # Real Batch Training
                def batch_train_step(input_data, target_data):
                    if args.relative_path:
                        raise Exception('No support for relative path now.')

                    grid_data = list()
                    for index in range(data_loader.batch_size):
                        grid_data.append(getSequenceGridMask(data_loader.norm_to_raw(input_data[index]),
                                                             args.neighborhood_size,
                                                             args.grid_size))
                    grid_data = np.stack(grid_data, axis=0)

                    feed = {model.input_data: input_data,
                            model.target_data: target_data,
                            model.grid_data: grid_data}

                    _train_loss, _, _counter = sess.run([model.cost, model.train_op, model.counter], feed)

                    return _train_loss, _counter

                train_loss, counter = batch_train_step(x, y)
                loss_batch += train_loss
                counter_batch += counter

                # For each sequence in the batch
                for batch in range(data_loader.batch_size):
                    # x_batch, y_batch would be numpy arrays of size seq_length x maxNumPeds x 3
                    x_batch, y_batch = x[batch], y[batch]

                    # Abandoned For Real Batch Training.
                    # raw_x_batch = data_loader.norm_to_raw(x_batch)
                    # grid_batch = getSequenceGridMask(raw_x_batch, args.neighborhood_size, args.grid_size)
                    #
                    # if args.relative_path:
                    #     raise Exception('No support for relative path now.')
                    # else:
                    #     input_data = x_batch
                    #     target_data = y_batch
                    #
                    # feed = {model.input_data: input_data,
                    #         model.target_data: target_data,
                    #         model.grid_data: grid_batch}
                    #
                    # train_loss, _, counter = sess.run([model.cost, model.train_op, model.counter], feed)
                    # loss_batch += train_loss
                    # counter_batch += counter

                    def get_full_counter(data):
                        # Get maximum edge of counter by the number of valid vrus.
                        full_counter = 0
                        for index in range(data.shape[0]):
                            if data[index, :].sum() != 0:
                                full_counter += data.shape[1]
                        return full_counter

                    full_counter_batch += get_full_counter(x_batch)
                    # assert counter <= get_full_counter(input_data)

                end = time.time()
                loss_batch = loss_batch / data_loader.batch_size

                # log and summary
                recorder.logger.info(
                    "{}/{} (epoch {}), train_loss = {:.3f}, time/batch = {:.3f}, cnt/full = {}/{}"
                        .format(
                        e * len(data_loader) + b,
                        args.num_epochs * len(data_loader),
                        e,
                        loss_batch, end - start,
                        counter_batch, full_counter_batch))
                recorder.writer.add_scalar('{}/train_loss'.format(args.phase), loss_batch, global_step=e)

                # Save the model if the current epoch and batch number match the frequency
                if (e * len(data_loader) + b) % args.save_every == 0 and ((e * len(data_loader) + b) > 0):
                    checkpoint_path = os.path.join(savepath, 'social_model.ckpt')
                    saver.save(sess, checkpoint_path, global_step=e * len(data_loader) + b,
                               write_meta_graph=False)
                    recorder.logger.info("model saved to {}".format(checkpoint_path))


if __name__ == '__main__':
    main()
