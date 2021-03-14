import tensorflow as tf
from tensorflow.keras.layers import Dense, ReLU


class Classifier(tf.keras.Model):
    def __init__(self):
        super(Classifier, self).__init__()
        self.dense1 = Dense(5)
        self.relu = ReLU()

    def call(self, inputs, training=None, mask=None):
        """
        :param inputs: A tensor or list of tensors.
        :param training: Boolean or boolean scalar tensor, indicating whether to run
          the `Network` in training mode or inference mode.
        :param mask: A mask or list of masks. A mask can be
            either a tensor or None (no mask).
        :return: A tensor if there is a single output, or
                a list of tensors if there are more than one outputs.
        """
        x = self.dense1(input)
        x = self.relu(x)
        return x
1