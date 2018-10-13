from tensorflow import keras, Tensor
import tensorflow as tf
from tensorflow.python.eager import context
from tensorflow.python.keras import Model


class TensorBoardImage(keras.callbacks.Callback):

    @staticmethod
    def make_image(tensor):
        """
        Convert an numpy representation image to Image protobuf.
        Copied from https://github.com/lanpa/tensorboard-pytorch/
        """
        from PIL import Image
        height, width, channel = tensor.shape
        image = Image.fromarray(tensor)
        import io
        output = io.BytesIO()
        image.save(output, format='PNG')
        image_string = output.getvalue()
        output.close()
        return tf.Summary.Image(height=height,
                                width=width,
                                colorspace=channel,
                                encoded_image_string=image_string)

    def __init__(self, logdir, tag, images):
        super().__init__()
        self.logdir = logdir
        self.tag = tag
        self.images = images
        self.sess = tf.Session()

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}
        # Do something to the image
        assert isinstance(self.model, Model)
        images = tf.convert_to_tensor(self.model.predict([[self.images[0]]]), dtype='float32')

        summary = tf.summary.image(self.tag, images, len(self.images))
        writer = tf.summary.FileWriter(logdir=self.logdir)
        writer.add_summary(summary.eval(session=self.sess), epoch)
        writer.close()

        return
