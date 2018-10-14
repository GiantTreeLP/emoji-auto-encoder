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
        super(CheckpointCallback, self).__init__(filepath,
                                                 monitor,
                                                 verbose,
                                                 save_best_only,
                                                 save_weights_only,
                                                 mode,
                                                 period)
        self.logger = ProgbarLogger()

    def on_epoch_begin(self, epoch, logs=None):
        self.logger.on_epoch_begin(epoch, logs)

    def on_epoch_end(self, epoch, logs=None):
        super().on_epoch_end(epoch, logs)
        if self.epochs_since_last_save == 0:
            open("../logs/epoch", "w").write(str(epoch))
            self.logger.on_epoch_end(epoch, logs)

    def on_train_begin(self, logs=None):
        self.logger.on_train_begin(logs)

    def on_train_end(self, logs=None):
        self.logger.on_epoch_end(logs)

    def on_batch_begin(self, batch, logs=None):
        self.logger.on_batch_begin(batch, logs)

    def on_batch_end(self, batch, logs=None):
        self.logger.on_batch_end(batch, logs)
