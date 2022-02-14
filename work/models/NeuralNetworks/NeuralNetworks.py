import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras import optimizers
from tensorflow.keras import layers
from tensorflow.keras import metrics
from tensorflow.keras import losses
from tensorflow.keras import initializers
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dropout, Dense
from tensorflow.keras import Input
from tensorflow import keras
from work.models.NeuralNetworks.Callbacks import *
import copy

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import mean_absolute_error

class NeuralNetwork(BaseEstimator, RegressorMixin):
    def __init__(self, name, model_):
        self.name = name
        self.model_ = model_
        self.N_OUTPUT = model_["N_OUTPUT"] 

        # Layers parameters (for the last layers in a CNN)
        self.neurons_per_layer = model_["neurons_per_layer"]
        self.activations = model_["activations"]
        
        self.batch_norm = model_["batch_norm"]
        self.batch_norm_epsilon = model_["batch_norm_epsilon"]
        self.dropout_rate = model_["dropout_rate"]                
        self.use_bias = model_["use_bias"]

        # Initializers
        self.default_kernel_initializer = copy.deepcopy(
            model_["default_kernel_initializer"])
        self.out_layer_kernel_initializer = copy.deepcopy(
            model_["out_layer_kernel_initializer"])
        self.default_bias_initializer = copy.deepcopy(
            model_["default_bias_initializer"])
        self.out_layer_bias_initializer = copy.deepcopy(
            model_["out_layer_bias_initializer"])

        # Regularizers
        self.default_activity_regularizer = copy.deepcopy(
            model_["default_activity_regularizer"])
        
        # Gradient descent parameters
        self.learning_rate = model_["learning_rate"]
        self.optimizer = model_["optimizer"]
        self.loss = model_["loss_function"]
        self.metrics = model_["metrics"]
        self.adapt_lr = model_["adapt_lr"]

        # Fit parameters
        self.n_epochs = model_["n_epochs"]
        self.batch_size = model_["batch_size"]
        self.early_stopping = model_["early_stopping"]
        self.early_stopping_alpha = model_["early_stopping_alpha"]
        self.shuffle_train = model_["shuffle_train"]

        self.stop_after = model_["stop_after"]
        self.stop_threshold = model_["stop_threshold"]

        # Spliter
        self.spliter = model_["spliter"]

    def set_params(self, **parameters):
        for parameter, value in parameters.items():            
            setattr(self, parameter, value)
        return self
    
    def fit(self, X, y, verbose=0):
        self.update_params()

        # Resplit the data
        ((X, y), (Xv, yv)) = self.spliter(X, y)
        self.model.fit(X, y, epochs=self.n_epochs, batch_size=self.batch_size,
                       callbacks=self.callbacks, validation_data=(Xv, yv),
                       shuffle=self.shuffle_train, verbose=verbose)
        
        return self
    
    def predict(self, X):
        return self.model.predict_step(X)
    
    def score(self, X, y):
        yhat = self.predict(X)
        return mean_absolute_error(y, yhat)        

    def update_params(self, input_shape=None):
        """
        Update the mlp's networks and the model's callback when a parameter changes
        """
        # Set callbacks
        self.callbacks = []
        self.early_stopping_callbacks()
        self.adapt_lr_callbacks()
        #self.tboard_callbacks()

        # Instantiate the model
        self.model = self.create_network(input_shape=input_shape)
        
    def create_network(self, input_shape=None):
        raise("Not Implemented by default!")

    def tboard_callbacks(self):
        self.callbacks.append(tboard())       
        
    def adapt_lr_callbacks(self):
        if self.adapt_lr:
            self.callbacks.append(
                ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                  patience=int(self.n_epochs/100),
                                  verbose=0, mode='min', min_delta=0.0,
                                  cooldown=0, min_lr=0.0001))
                
    def early_stopping_callbacks(self):
        if self.early_stopping:
            alpha = self.early_stopping_alpha
            if self.early_stopping == "best_epoch":
                self.callbacks.append(EarlyStoppingBestEpoch(
                    monitor='val_loss', verbose=0))
                
            if self.early_stopping == "decrease_val_loss":
                self.callbacks.append(EarlyStoppingDecreasingValLoss(
                    monitor='val_loss', patience=int(self.n_epochs/10),
                    verbose=0, restore_best_weights=True))
                
            if (self.early_stopping == "sliding_average"
                or self.early_stopping == "both"):
                self.callbacks.append(EarlyStoppingSlidingAverage(
                    monitor = 'val_loss', patience=alpha,
                    verbose=0, alpha=alpha, restore_best_weights=True))
                
            if (self.early_stopping == "prechelt"
                or self.early_stopping == "both"):
                self.callbacks.append(PrecheltEarlyStopping(
                    monitor = 'loss', val_monitor = 'val_loss',
                    baseline = 2.5, verbose=0, alpha=alpha))

            if self.stop_after > 0:
                if self.stop_threshold > 0:
                    self.callbacks.append(TimeStoppingAndThreshold(
                        seconds=self.stop_after, verbose=0,
                        threshold=self.stop_threshold,
                        monitor="val_loss"))
                else:
                    self.callbacks.append(tfa.callbacks.TimeStopping(
                        seconds=self.stop_after, verbose=0))
                                          
