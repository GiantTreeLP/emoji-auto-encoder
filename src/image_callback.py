import io

import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow import keras


class TensorBoardImage(keras.callbacks.TensorBoard):

    def __init__(self, log_dir, tag, images, period=1):
        super(TensorBoardImage, self).__init__(log_dir=log_dir)
        self.tag = tag
        self.images = images
        self.period = period
        self.last_save = 0

    def on_train_begin(self, logs=None):
        super(TensorBoardImage, self).on_train_begin(logs)
        img_bytes = io.BytesIO()
        original = Image.fromarray(np.reshape(self.images[0] * 255, (128, 128)).astype('uint8'), 'L')
        original.save(img_bytes, 'png')
        original = tf.Summary.Image(width=original.width, height=original.height,
                                    encoded_image_string=img_bytes.getvalue())
        self.writer.add_summary(tf.Summary(value=[tf.Summary.Value(tag="Original", image=original)]))

    def on_epoch_end(self, epoch, logs=None):
        super(TensorBoardImage, self).on_epoch_end(epoch, logs)

        self.last_save += 1
        if self.last_save >= self.period:
            self.last_save = 0

            prediction = self.model.predict([[self.images[0]]])[0]
            prediction = prediction * 255
            prediction = prediction.astype('uint8')
            prediction = np.reshape(prediction, (128, 128))
            img_bytes = io.BytesIO()
            image = Image.fromarray(prediction, "L")
            image.save(img_bytes, format="png")
            image = tf.Summary.Image(height=image.height, width=image.width, encoded_image_string=img_bytes.getvalue())
            summary = tf.Summary(value=[tf.Summary.Value(tag=self.tag, image=image)])

            self.writer.add_summary(summary, epoch)
