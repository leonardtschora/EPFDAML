import tensorflow as tf
from tensorflow.keras.callbacks import Callback
import warnings
from tensorflow.keras import backend as bk
import numpy as np
from datetime import datetime
import time

def tboard(logdir="logs"):
    logs = "logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    return tf.keras.callbacks.TensorBoard(log_dir = logdir,
                                          histogram_freq = 1,
                                          profile_batch = '500,520')

def GL(t, val_monitored):
    epsilon = 10e-7
    Eva_t = val_monitored[t]
    if t == 0:
        Eopt_t = val_monitored[0]
    else:
        Eopt_t = min(val_monitored[0:t])
    return 100 * (Eva_t / (Eopt_t + epsilon) -1)


def Pk(t, monitored, k):
    epsilon = 10e-7
    if t < k:
        return 1
    else:
        strip = monitored[t - k:t]
        return 1000 * (np.mean(strip) / (min(strip) + epsilon) - 1)


def custom_early_stopping(t, k, values_train, values_val):
    epsilon = 10e-7
    return GL(t, values_val) / (Pk(t, values_train, k) + epsilon)
        

class PrecheltEarlyStopping(Callback):
    """Stop training when the mean monitored quantity has stopped decreasing
    # Arguments
        monitor: quantity to be monitored.
        val_monitor: validation quantity to be monitored
        alpha: strip length
        verbose: verbosity mode.
        baseline: Baseline value for the monitored quantity to reach.
            Training will stop if the model doesn't show improvement
            over the baseline.
    """

    def __init__(self, monitor='loss', val_monitor = 'val_loss', baseline=3, verbose=0, alpha=10):
        super(PrecheltEarlyStopping, self).__init__()

        self.monitor = monitor
        self.val_monitor = val_monitor
        self.verbose = verbose
        self.alpha = alpha
        self.baseline = baseline
        self.stopped_epoch = 0
        self.best_weights = None

    def on_train_begin(self, logs=None):
        # Allow instances to be re-used
        self.wait = 0
        self.stopped_epoch = 0
        self.val_losses = []
        self.losses = []
        self.best = self.baseline

    def on_epoch_end(self, epoch, logs=None):
        current = self.get_monitor_value(logs)
        if current is None:
            return

        if current > self.baseline:
            self.stopped_epoch = epoch
            self.model.stop_training = True

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0 and self.verbose > 0:
            print('Epoch %05d: early stopping' % (self.stopped_epoch + 1))
        self.best_epoch = self.stopped_epoch
        
    def get_monitor_value(self, logs):
        monitor_value_train = logs.get(self.monitor)
        if monitor_value_train is None:
            warnings.warn(
                'Early stopping conditioned on metric `%s` '
                'which is not available. Available metrics are: %s' %
                (self.monitor, ','.join(list(logs.keys()))), RuntimeWarning
            )

        self.losses.append(monitor_value_train)

        monitor_value_val = logs.get(self.val_monitor)
        if monitor_value_train is None:
            warnings.warn(
                'Early stopping conditioned on metric `%s` '
                'which is not available. Available metrics are: %s' %
                (self.val_monitor, ','.join(list(logs.keys()))), RuntimeWarning
            )

        self.val_losses.append(monitor_value_val)
        
        try:
            current = len(self.val_losses)
        except:
            current = 0
            values_train = [self.losses[current]]
            values_val = [self.val_losses[current]]
        else:
            values_val = self.val_losses
            values_train = self.losses

        if current < self.alpha + 1:
            return self.baseline - 1
        else:
            criteria = custom_early_stopping(current - 1, self.alpha, values_train, values_val)
            return criteria
    

class EarlyStoppingSlidingAverage(Callback):
    """Stop training when the mean monitored quantity has stopped decreasing
    # Arguments
        monitor: quantity to be monitored.
        patience: number of epochs with no improvement
            after which training will be stopped.
        alpha: number of epochs to take in account to compute the mean
        verbose: verbosity mode.
        restore_best_weights: whether to restore model weights from
            the epoch with the best value of the monitored quantity.
            If False, the model weights obtained at the last step of
            training are used.
    """

    def __init__(self, monitor='val_loss', patience=0, verbose=0, alpha=10, restore_best_weights=False):
        super(EarlyStoppingSlidingAverage, self).__init__()

        self.monitor = monitor
        self.patience = patience
        self.verbose = verbose
        self.alpha = alpha
        self.wait = 0
        self.stopped_epoch = 0
        self.restore_best_weights = restore_best_weights
        self.best_weights = None

    def on_train_begin(self, logs=None):
        # Allow instances to be re-used
        self.wait = 0
        self.stopped_epoch = 0
        self.val_losses = []
        self.best = 10e9

    def on_epoch_end(self, epoch, logs=None):
        current = self.get_monitor_value(logs)
        if current is None:
            return

        if current < self.best:
            self.best = current
            self.wait = 0
            if self.restore_best_weights:
                self.best_weights = self.model.get_weights()
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True
                if self.restore_best_weights:
                    if self.verbose > 0:
                        print('Restoring model weights from the end of '
                              'the best epoch')
                    self.model.set_weights(self.best_weights)

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0 and self.verbose > 0:
            print('Epoch %05d: early stopping' % (self.stopped_epoch + 1))
  
    def get_monitor_value(self, logs):
        monitor_value = logs.get(self.monitor)
        if monitor_value is None:
            print(
                'Early stopping conditioned on metric `%s` '
                'which is not available. Available metrics are: %s' %
                (self.monitor, ','.join(list(logs.keys()))), RuntimeWarning
            )

        self.val_losses.append(monitor_value)
        try:
            current = len(self.val_losses)
        except:
            current = 0
            values = [self.val_losses[current]]
        else:
            k = min(self.alpha, current)
            values = self.val_losses[current - k:current]
            
        mean_ = np.mean(values)
        return mean_


class EarlyStoppingBestEpoch(Callback):
    """Train the model on all specified epochs. At the end of training, restore best weights
    # Arguments
        monitor: quantity to be monitored.
        verbose: verbosity mode.
    """

    def __init__(self, monitor='val_loss', verbose=0):
        super(EarlyStoppingBestEpoch, self).__init__()

        self.monitor = monitor
        self.verbose = verbose
        self.best_weights = None

    def on_train_begin(self, logs=None):
        # Allow instances to be re-used
        self.stopped_epoch = 0
        self.best_epoch = 0
        self.best = 10e9

    def on_epoch_end(self, epoch, logs=None):
        current = self.get_monitor_value(logs)
        if current is None:
            return

        if current < self.best:
            self.best_epoch = epoch
            self.best = current
            self.best_weights = self.model.get_weights()
                        

    def on_train_end(self, logs=None):
        if self.verbose > 0:
            print('Restoring model weights from the end of '
                  'the best epoch %05d' % (self.stopped_epoch + 1))
        self.model.set_weights(self.best_weights)
  
    def get_monitor_value(self, logs):
        monitor_value = logs.get(self.monitor)
        if monitor_value is None:
            warnings.warn(
                'Early stopping conditioned on metric `%s` '
                'which is not available. Available metrics are: %s' %
                (self.monitor, ','.join(list(logs.keys()))), RuntimeWarning
            )

        return monitor_value


class EarlyStoppingDecreasingValLoss(Callback):
    """Stop training when the monitored quantity has stopped decreasing
    # Arguments
        monitor: quantity to be monitored.
        patience: number of epochs with no improvement
            after which training will be stopped.
        verbose: verbosity mode.
        restore_best_weights: whether to restore model weights from
            the epoch with the best value of the monitored quantity.
            If False, the model weights obtained at the last step of
            training are used.
    """

    def __init__(self, monitor='val_loss', patience=0, verbose=0, restore_best_weights=False):
        super(EarlyStoppingDecreasingValLoss, self).__init__()

        self.monitor = monitor
        self.patience = patience
        self.verbose = verbose
        self.wait = 0
        self.stopped_epoch = 0
        self.restore_best_weights = restore_best_weights
        self.best_weights = None

    def on_train_begin(self, logs=None):
        # Allow instances to be re-used
        self.wait = 0
        self.stopped_epoch = 0
        self.best_epoch = 0
        self.val_losses = []
        self.best = 10e9

    def on_epoch_end(self, epoch, logs=None):
        current = self.get_monitor_value(logs)
        if current is None:
            return

        if current < self.best:
            self.best = current
            self.best_epoch = epoch
            self.wait = 0
            if self.restore_best_weights:
                self.best_weights = self.model.get_weights()
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True
                if self.restore_best_weights:
                    if self.verbose > 0:
                        print('Restoring model weights from the end of '
                              'the best epoch')
                    self.model.set_weights(self.best_weights)

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0 and self.verbose > 0:
            print('Epoch %05d: early stopping' % (self.stopped_epoch + 1))
  
    def get_monitor_value(self, logs):
        monitor_value = logs.get(self.monitor)
        if monitor_value is None:
            warnings.warn(
                'Early stopping conditioned on metric `%s` '
                'which is not available. Available metrics are: %s' %
                (self.monitor, ','.join(list(logs.keys()))), RuntimeWarning
            )
        return monitor_value

    
class TimeStoppingAndThreshold(Callback):
    """Stop training when a specified amount of time has passed
    AND the specified metric is still higher than a specified threshold
    Args:
        seconds: maximum amount of time before stopping.
            Defaults to 86400 (1 day).
    
        verbose: verbosity mode. Defaults to 0.
    """

    def __init__(self, monitor="val_loss", seconds: int = 86400,
                 verbose: int = 0, threshold = 100, alpha=25):
        super().__init__()

        self.seconds = seconds
        self.verbose = verbose
        self.stopped_epoch = None

        self.monitor = monitor
        self.alpha = alpha
        self.threshold = threshold

    def on_train_begin(self, logs=None):
        self.continue_training = False
        self.stopping_time = time.time() + self.seconds
        self.stopped_epoch = 0
        self.val_losses = []

    def get_monitor_value(self, logs):
        monitor_value = logs.get(self.monitor)
        if monitor_value is None:
            print(
                'Early stopping conditioned on metric `%s` '
                'which is not available. Available metrics are: %s' %
                (self.monitor, ','.join(list(logs.keys()))), RuntimeWarning
            )

        self.val_losses.append(monitor_value)
        try:
            current = len(self.val_losses)
        except:
            current = 0
            values = [self.val_losses[current]]
        else:
            k = min(self.alpha, current)
            values = self.val_losses[current - k:current]
            
        mean_ = np.mean(values)
        return mean_
            
    def on_epoch_end(self, epoch, logs=None):
        """
        If the time limit has been reached but the error is lower 
        than the threshold, disable this callback
        """
        if not self.continue_training:
            if (time.time() >= self.stopping_time):
                current = self.get_monitor_value(logs)
                if current > self.threshold:
                    self.model.stop_training = True
                    self.stopped_epoch = epoch
                else:
                    self.continue_training = True

    def on_train_end(self, logs=None):
        if self.stopped_epoch is not None and self.verbose > 0:
            formatted_time = datetime.timedelta(seconds=self.seconds)
            msg = "Timed stopping at epoch {} after training for {}".format(
                self.stopped_epoch + 1, formatted_time
            )
            print(msg)
