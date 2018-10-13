import io

import tensorflow as tf
from PIL import Image
from tensorflow import keras


class TensorBoardImage(keras.callbacks.Callback):

    def __init__(self, logdir, tag, images):
        super().__init__()
        self.logdir = logdir
        self.tag = tag
        self.images = images
        cfg = tf.ConfigProto()
        self.sess = tf.Session(config=cfg)
        self.writer = tf.summary.FileWriter(logdir=self.logdir, session=self.sess)

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}
        with tf.device("/cpu:0"):
            prediction = self.model.predict([[self.images[0]]])[0]
            prediction = prediction * 255
            prediction = prediction.astype('uint8')
            img_bytes = io.BytesIO()
            image = Image.fromarray(prediction)
            image.save(img_bytes, format="png")
            image = tf.Summary.Image(height=image.height, width=image.width, encoded_image_string=img_bytes.getvalue())
            summary = tf.Summary(value=[tf.Summary.Value(tag=self.tag, image=image)])

        self.writer.add_summary(summary, epoch)
        self.writer.flush()

    def on_train_end(self, logs=None):
        self.writer.flush()
        self.writer.close()
        pass
