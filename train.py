import tensorflow as tf
import argparse
import os
import time
import pickle
import logging
import sys
import numpy as np

from model import Model
from kitti_utils import KittiDataLoader

FORMAT = '[%(levelname)s: %(filename)s: %(lineno)4d]: %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT, stream=sys.stdout)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    # RNN size parameter (dimension of the output/hidden state)
    parser.add_argument('--rnn_size', type=int, default=128,
                        help='size of RNN hidden state')
    # Number of layers parameter
    # TODO: (improve) Number of layers not used. Only a single layer implemented
    parser.add_argument('--num_layers', type=int, default=1,
                        help='number of layers in the RNN')
    # Type of recurrent unit parameter
    # Model currently not used. Only LSTM implemented
    parser.add_argument('--model', type=str, default='lstm',
                        help='rnn, gru, or lstm')
    # Size of each batch parameter
    parser.add_argument('--batch_size', type=int, default=256,
                        help='minibatch size')
    # Length of sequence to be considered parameter
    parser.add_argument('--seq_length', type=int, default=5,
                        help='RNN sequence length')
    # Number of epochs parameter
    parser.add_argument('--num_epochs', type=int, default=500,
                        help='number of epochs')
    # Frequency at which the model should be saved parameter (单位-epochs)
    parser.add_argument('--save_every', type=int, default=50,
                        help='save frequency')
    # Gradient value at which it should be clipped
    # TODO: (resolve) Clipping gradients for now. No idea whether we should
    parser.add_argument('--grad_clip', type=float, default=10.,
                        help='clip gradients at this value')
    # Learning rate parameter
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='learning rate')
    # Decay rate for the learning rate parameter
    parser.add_argument('--decay_rate', type=float, default=0.99,
                        help='decay rate for rmsprop')
    # Dropout probability parameter
    # Dropout not implemented.
    parser.add_argument('--keep_prob', type=float, default=0.8,
                        help='dropout keep probability')
    # Dimension of the embeddings parameter
    parser.add_argument('--embedding_size', type=int, default=64,
                        help='Embedding dimension for the spatial coordinates')
    # save path
    parser.add_argument('--save_path', type=str, default='vanilla-lstm-save')
    # 是否采用相对型作为输入
    parser.add_argument('--relative_path', type=bool, default=True,
                        help='Use relative path as obs and pred, default True')
    args = parser.parse_args()
    logger.info(args)
    train(args)


def train(args):
    # Create the data loader object. This object would preprocess the data in terms of
    # batches each of size args.batch_size, of length args.seq_length
    # data_loader = (args.batch_size, args.seq_length, datasets)
    data_loader = KittiDataLoader(args.batch_size, args.seq_length, [], sub_set='train')

    # Save the arguments int the config file
    if os.path.exists(args.save_path):
        x = input('Save exist, continue? [y/n]?')
        if x != 'y':
            exit(0)
    else:
        os.mkdir(args.save_path)
    with open(os.path.join(args.save_path, 'config.pkl'), 'wb') as f:
        pickle.dump(args, f)

    # Create a Vanilla LSTM model with the arguments
    model = Model(args)

    # Initialize a TensorFlow session
    with tf.Session() as sess:
        # Initialize all the variables in the graph
        sess.run(tf.initialize_all_variables())
        # Add all the variables to the list of variables to be saved
        saver = tf.train.Saver(tf.all_variables())

        # For each epoch
        for e in range(args.num_epochs):
            # Assign the learning rate (decayed acc. to the epoch number)
            # yzn 每一个epoch后降低学习率的大小
            sess.run(tf.assign(model.lr, args.learning_rate * (args.decay_rate ** e)))
            # Reset the pointers in the data loader object
            data_loader.reset_ptr()
            # Get the initial cell state of the LSTM
            state = sess.run(model.initial_state)

            # For each batch in this epoch
            for b in range(len(data_loader)):
                # Tic
                start = time.time()
                # Get the source and target data of the current batch
                # x has the source data, y has the target data
                x, y = data_loader.next_batch()

                input_x = np.zeros((args.batch_size, args.seq_length, 4))
                input_y = np.zeros((args.batch_size, args.seq_length, 4))

                input_x[:, :, 0], input_y[:, :, 0] = x[:, :, 8], y[:, :, 8]  # abs location x
                input_x[:, :, 1], input_y[:, :, 1] = x[:, :, 10], y[:, :, 10]  # abs location y
                input_x[:, :, 2], input_y[:, :, 2] = x[:, :, 3] - x[:, :, 1], y[:, :, 3] - y[:, :, 1]  # bb width
                input_x[:, :, 3], input_y[:, :, 3] = x[:, :, 4] - x[:, :, 2], y[:, :, 4] - y[:, :, 2]  # bb height

                def abs_to_rel(x):
                    result = np.zeros_like(x)
                    result[:, 0, 0] = 0
                    result[:, 0, 1] = 0
                    result[:, :, 2] = x[:, :, 2]
                    result[:, :, 3] = x[:, :, 3]
                    for i in range(1, result.shape[1]):
                        result[:, i, 0] = x[:, i, 0] - x[:, i - 1, 0]
                        result[:, i, 1] = x[:, i, 1] - x[:, i - 1, 1]
                    return result

                # print('input_x_rel:', input_x[0])
                # print('input_y_rel:', input_y[0])
                if args.relative_path:
                    input_x = abs_to_rel(input_x)
                    input_y = abs_to_rel(input_y)
                # print('input_x_rel:', input_x[0])
                # print('input_y_rel:', input_y[0])

                # Feed the source, target data and the initial LSTM state to the model
                feed = {model.input_data: input_x, model.target_data: input_y, model.initial_state: state}
                # Fetch the loss of the model on this batch, the final LSTM state from the session
                train_loss, state, _ = sess.run([model.cost, model.final_state, model.train_op], feed)
                # Toc
                end = time.time()
                # Print epoch, batch, loss and time taken
                print(
                    "{}/{} (epoch {}), train_loss = {:.3f}, time/batch = {:.3f}"
                        .format(
                        e * len(data_loader) + b,
                        args.num_epochs * len(data_loader),
                        e,
                        train_loss, end - start))

                # Save the model if the current epoch and batch number match the frequency
                # if (e * len(data_loader) + b) % args.save_every == 0 and ((e * len(data_loader) + b) > 0):
                if (e + 1) % args.save_every == 0 and b == 0:
                    checkpoint_path = os.path.join(args.save_path, 'model.ckpt')
                    saver.save(sess, checkpoint_path, global_step=e * len(data_loader) + b)
                    print("model saved to {}".format(checkpoint_path))


if __name__ == '__main__':
    main()
