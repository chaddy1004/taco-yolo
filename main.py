import tensorflow as tf
import numpy as np
import random

from trainer.sl_trainer import SLTrainer

tf.random.set_seed(19971124)
np.random.seed(100)
random.seed(13)

if __name__ == '__main__':
    trainer = SLTrainer()
    trainer.train()