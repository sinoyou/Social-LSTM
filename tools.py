import logging
import sys
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import os

confidence = 5.991  # it's not used for plotting potential zone.
ellipse_args = {'ec': 'blue', 'fill': False, 'lw': 1, 'alpha': 0.5}
plot_args = {'lw': 2, 'alpha': 0.5, 'marker': '*'}
patch_args = {'alpha': 0.9}


class Recorder:
    """
    Designed specially for recording multiple type logging information.
    """

    def __init__(self, summary_path='default', board=True, logfile=True, stream=True):
        """
        :param summary_path: path for saving summary and log file.
        :param board: T/F, if need summary writer.
        :param logfile: T/F, if need to generate log file.
        :param stream: T/F, if need to show
        """
        saved_summary_filepath = '{}/'.format(summary_path)
        if not os.path.exists(saved_summary_filepath):
            os.makedirs(saved_summary_filepath)
        # board
        if board:
            self.writer = SummaryWriter(saved_summary_filepath)
        else:
            self.writer = None
        # log info
        FORMAT = '[%(levelname)s %(asctime)s: %(filename)s: %(lineno)4d]: %(message)s'
        datefmt = '[%Y-%m-%d %H:%M:%S]'
        self.logger = logging.getLogger(name=saved_summary_filepath)
        file_handler = logging.FileHandler(filename=os.path.join(saved_summary_filepath, 'runner.log'))
        stream_handler = logging.StreamHandler(stream=sys.stdout)

        formatter = logging.Formatter(FORMAT, datefmt)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        stream_handler.setLevel(logging.INFO)
        stream_handler.setFormatter(formatter)

        if stream:
            self.logger.addHandler(stream_handler)
        if logfile:
            self.logger.addHandler(file_handler)
        self.logger.setLevel(logging.INFO)

    def close(self):
        if self.writer:
            self.writer.close()