import io

import tensorflow as tf
from PIL import Image
from tensorflow import keras


class TensorBoardImage(keras.callbacks.TensorBoard):

    def __init__(self, log_dir, tag, images):
        super().__init__(log_dir=log_dir)
        self.tag = tag
        self.images = images
        cfg = tf.ConfigProto()
        self.sess = tf.Session(config=cfg)

    def on_epoch_end(self, epoch, logs=None):
        super(TensorBoardImage, self).on_epoch_end(epoch, logs)
        if logs is None:
            logs = {}
        prediction = self.model.predict([[self.images[0]]])[0]
        prediction = prediction * 255
        prediction = prediction.astype('uint8')
        img_bytes = io.BytesIO()
        image = Image.fromarray(prediction)
        image.save(img_bytes, format="png")
        image = tf.Summary.Image(height=image.height, width=image.width, encoded_image_string=img_bytes.getvalue())
        summary = tf.Summary(value=[tf.Summary.Value(tag=self.tag, image=image)])

        self.writer.add_summary(summary, epoch)