import numpy as np
import tensorflow as tf
from tensorflow import keras


class TensorBoardImage(keras.callbacks.Callback):

    def __init__(self, log_dir, tag, images, period=1):
        super().__init__()
        self.log_dir = log_dir
        self.tag = tag
        self.sample_image = np.reshape(images[0:3], (-1, 128, 128, 4))
        self.period = period
        self.last_save = 0
        self.writer = tf.summary.create_file_writer(self.log_dir)

    def on_train_begin(self, logs=None):
        with self.writer.as_default():
            tf.summary.image(name="Original", data=self.sample_image, step=0)

    def on_epoch_end(self, epoch, logs=None):
        super(TensorBoardImage, self).on_epoch_end(epoch, logs)

        self.last_save += 1
        if self.last_save >= self.period:
            self.last_save = 0

            prediction = self.model.predict(self.sample_image)
            prediction = np.reshape(prediction, (-1, 128, 128, 4))

            with self.writer.as_default():
                tf.summary.image(name=self.tag, data=prediction, step=epoch, max_outputs=len(self.sample_image))
