from tensorflow.python.keras.callbacks import ModelCheckpoint


class CheckpointCallback(ModelCheckpoint):
    def __init__(self,
                 filepath,
                 monitor='val_loss',
                 verbose=0,
                 save_best_only=False,
                 save_weights_only=False,
                 mode='auto',
                 period=1):
        super(CheckpointCallback, self).__init__(filepath, monitor, verbose, save_best_only, save_weights_only, mode,
                                                 period)

    def on_epoch_end(self, epoch, logs=None):
        super().on_epoch_end(epoch, logs)
        if self.epochs_since_last_save == 0:
            open("../logs/epoch", "w").write(str(epoch))
