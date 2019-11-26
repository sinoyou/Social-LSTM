import tensorflow as tf
import argparse
import os
import time
import pickle
import numpy as np
import logging
import sys

import path_static
from social_model import SocialModel
from deprecated.social_utils_mot16 import SocialDataLoader
from grid import getSequenceGridMask
from social_utils_kitti import KittiDataLoader

FORMAT = '[%(levelname)s: %(filename)s: %(lineno)4d]: %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT, stream=sys.stdout)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
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
    parser.add_argument('--batch_size', type=int, default=64,
                        help='minibatch size')
    # Length of sequence to be considered parameter
    parser.add_argument('--seq_length', type=int, default=10,
                        help='RNN sequence length')
    # Number of epochs parameter
    parser.add_argument('--num_epochs', type=int, default=30,
                        help='number of epochs')
    # Frequency at which the model should be saved parameter
    parser.add_argument('--save_every', type=int, default=50,
                        help='save frequency')
    # TODO: (resolve) Clipping gradients for now. No idea whether we should
    # Gradient value at which it should be clipped
    parser.add_argument('--grad_clip', type=float, default=10.,
                        help='clip gradients at this value')
    # Learning rate parameter
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='learning rate')
    # Decay rate for the learning rate parameter
    parser.add_argument('--decay_rate', type=float, default=0.95,
                        help='decay rate for rmsprop')
    # Dropout not implemented.
    # Dropout probability parameter
    parser.add_argument('--keep_prob', type=float, default=0.8,
                        help='dropout keep probability')
    # Dimension of the embeddings parameter
    parser.add_argument('--embedding_size', type=int, default=128,
                        help='Embedding dimension for the spatial coordinates')
    # Size of neighborhood to be considered parameter
    parser.add_argument('--neighborhood_size', type=int, default=5,
                        help='Neighborhood size to be considered for social grid')
    # Size of the social grid parameter
    parser.add_argument('--grid_size', type=int, default=2,
                        help='Grid size of the social grid')
    # Maximum number of pedestrians to be considered
    parser.add_argument('--maxNumPeds', type=int, default=70,
                        help='Maximum Number of Pedestrians')
    # Image info
    parser.add_argument('--image_dim', type=tuple, default=(1392, 512),
                        help='image width and height for social pooling')
    # Save place
    parser.add_argument('--save_dir', type=str, help='directory of saving ckpt, log and config.')
    # Visible Device
    parser.add_argument('--device', type=str, default='0', help='GPU device num')
    args = parser.parse_args()
    logger.info(args)
    train(args)


def train(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device
    # Create the SocialDataLoader object
    data_loader = KittiDataLoader(args.batch_size, args.seq_length, args.maxNumPeds, ignore_list=[], sub_set='train')

    savepath = args.save_dir
    # save path check 当保存目录已经存在时需要特别处理，以防止保存的模型出现覆盖
    if os.path.exists(savepath):
        print("[WARNING]: Save Path Already Exists. Do you want to continue ? ")
        command = input('[y/n]:')
        if len(command) == 1 and command.lower()[0] == 'y':
            pass
        else:
            exit(0)
    else:
        os.mkdir(savepath)

    # Initialize a TensorFlow session
    with tf.Session() as sess:
        writer = tf.summary.FileWriter(logdir=os.path.join(args.save_dir, 'runs'), graph=sess.graph)

        # 模型初始化或预加载
        def get_model(force):
            if not force and os.path.exists(os.path.join(savepath, 'social_config.pkl')):
                with open(os.path.join(savepath, 'social_config.pkl'), 'rb') as f:
                    save_args = pickle.load(f)
                model = SocialModel(save_args)
                # Restore variables from checkpoint
                ckpt = tf.train.get_checkpoint_state(savepath)
                logger.info('loading model: ', ckpt.model_checkpoint_path)
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
                x, y, _ = data_loader.next_batch()

                # variable to store the loss for this batch
                loss_batch = 0

                # For each sequence in the batch
                for batch in range(data_loader.batch_size):
                    # x, y: batch_size x seq_length x maxNumPeds x 11
                    # x_batch, y_batch would be numpy arrays of size seq_length x maxNumPeds x 5
                    x_batch, y_batch = x[batch], y[batch]

                    def data_filter(data):
                        use_shape = (data.shape[0], data.shape[1], 5)
                        use_data = np.zeros(use_shape)
                        use_data[:, :, 0] = data[:, :, 0]
                        use_data[:, :, 1] = data[:, :, 8]  # original x -> x
                        use_data[:, :, 2] = data[:, :, 10]  # original z -> y
                        use_data[:, :, 3] = (data[:, :, 3] - data[:, :, 1])  # width
                        use_data[:, :, 4] = (data[:, :, 4] - data[:, :, 2])  # height
                        return use_data

                    use_x_batch = data_filter(x_batch)
                    use_y_batch = data_filter(y_batch)

                    grid_batch = getSequenceGridMask(x_batch, args.neighborhood_size, args.grid_size)

                    # Feed the source, target data
                    # use_x_batch -> use_x_relative_batch
                    # use_y_bacth -> use_y_relative_bacth
                    def abs_to_def(data):
                        result = np.zeros_like(data)
                        result[:, :, 0] = data[:, :, 0]  # id
                        result[:, :, 3:5] = data[:, :, 3:5]  # width, height
                        result[0, :, 1:3] = 0  # rel start
                        for i in range(1, data.shape[0]):
                            result[i, :, 1:3] = data[i, :, 1:3] - data[i - 1, :, 1:3]
                        return result

                    use_x_rel_batch = abs_to_def(use_x_batch)  # id, rel_x, rel_y, width, height
                    use_y_rel_batch = abs_to_def(use_y_batch)

                    feed = {model.input_data: use_x_rel_batch,
                            model.target_data: use_y_rel_batch,
                            model.grid_data: grid_batch}

                    train_loss, _, summary = sess.run([model.cost, model.train_op, model.merge], feed)

                    writer.add_summary(summary, global_step=e * len(data_loader) + b)

                    loss_batch += train_loss

                end = time.time()
                loss_batch = loss_batch / data_loader.batch_size
                logger.info(
                    "{}/{} (epoch {}), train_loss = {:.3f}, time/batch = {:.3f}"
                        .format(
                        e * len(data_loader) + b,
                        args.num_epochs * len(data_loader),
                        e,
                        loss_batch, end - start))

                # Save the model if the current epoch and batch number match the frequency
                if (e * len(data_loader) + b) % args.save_every == 0 and ((e * len(data_loader) + b) > 0):
                    checkpoint_path = os.path.join(savepath, 'social_model.ckpt')
                    saver.save(sess, checkpoint_path, global_step=e * len(data_loader) + b,
                               write_meta_graph=False)
                    logger.info("model saved to {}".format(checkpoint_path))


if __name__ == '__main__':
    main()
