import tensorflowjs as tfjs
from tensorflow.python.keras.callbacks import ModelCheckpoint, ProgbarLogger


class CheckpointCallback(ModelCheckpoint):
    def __init__(self,
                 filepath,
                 monitor='val_loss',
                 verbose=0,
                 save_best_only=False,
                 save_weights_only=False,
                 mode='auto',
                 period=1):
        super(CheckpointCallback, self).__init__(filepath + "/model.h5",
                                                 monitor,
                                                 verbose,
                                                 save_best_only,
                                                 save_weights_only,
                                                 mode,
                                                 period)
        self.logger = ProgbarLogger()
        self.logger.verbose = True
        self.dir = filepath

    def on_epoch_begin(self, epoch, logs=None):
        super(CheckpointCallback, self).on_epoch_begin(epoch, logs)
        if self.epochs_since_last_save == 0:
            self.logger.on_epoch_begin(epoch, logs)

    def on_epoch_end(self, epoch, logs=None):
        super().on_epoch_end(epoch, logs)
        if self.epochs_since_last_save == 0:
            open(self.dir + "/epoch.txt", "w").write(str(epoch))
            tfjs.converters.save_keras_model(self.model, self.dir)
            self.logger.on_epoch_end(epoch, logs)

    def on_train_begin(self, logs=None):
        super(CheckpointCallback, self).on_train_begin(logs)
        self.logger.on_train_begin(logs)

    def on_train_end(self, logs=None):
        super(CheckpointCallback, self).on_train_end(logs)
        self.logger.on_train_end(logs)
        tfjs.converters.save_keras_model(self.model, self.dir)

    def on_batch_begin(self, batch, logs=None):
        super(CheckpointCallback, self).on_batch_begin(batch)
        if self.epochs_since_last_save == 0:
            self.logger.on_batch_begin(batch, logs)

    def on_batch_end(self, batch, logs=None):
        super(CheckpointCallback, self).on_batch_end(batch)
        if self.epochs_since_last_save == 0:
            self.logger.on_batch_end(batch, logs)

    def set_model(self, model):
        super(CheckpointCallback, self).set_model(model)
        self.logger.set_model(model)

    def set_params(self, params):
        super(CheckpointCallback, self).set_params(params)
        params['verbose'] = True
        self.logger.set_params(params)
