from work.models.model_wrapper import *
from work.models.Splitter import MySplitter
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import BaggingRegressor
from tensorflow.keras import initializers
from tensorflow.keras import regularizers
from tensorflow.keras.models import load_model
import work.parallel_scikit as ps
import tensorflow.keras.backend as K

class NeuralNetWrapper(ModelWrapper):
    def __init__(self, prefix, dataset_name, label, spliter=None):
        ModelWrapper.__init__(self, prefix, dataset_name, label, spliter=spliter)
        if self.spliter is None: self.spliter = MySplitter(0.25)

    def params(self):
        return {
            "N_OUTPUT" : len(self.label),
    
            "batch_norm" : False,
            "batch_norm_epsilon" : 10e-7,

            "neurons_per_layer" : (240, 160, ),
            "activations" : ("relu", ),            
            "dropout_rate" : 0.25,
    
            "learning_rate" : 0.001,
            "optimizer" : "Adam",
            "loss_function" : "LogCosh",
            "metrics" : ["mean_absolute_error"],
            "use_bias" : True,
            
            "default_kernel_initializer" : initializers.he_uniform(),
            "out_layer_kernel_initializer" : initializers.lecun_uniform(),
            "default_bias_initializer" : initializers.Zeros(),
            "out_layer_bias_initializer" : initializers.Zeros(),

            "default_activity_regularizer" : None,    
            
            "adapt_lr" : False,        
            "n_epochs" : 100000,
            "batch_size" : 300,
            "early_stopping" : "sliding_average",
            "early_stopping_alpha" : 25,
            "stop_after" : -1,
            "stop_threshold": -1,
            "shuffle_train" : True,
            "scaler" : "BCM",
            "transformer" : "Standard",

            "spliter" : self.spliter,
        }

    def map_dict(self):
        orig = ModelWrapper.map_dict(self)
        orig.update({
            "default_activity_regularizer" :
            (mu.activity_regularizer_to_string,
             mu.activity_regularizer_from_string),
            "neurons_per_layer" :
            (mu.neurons_per_layer_to_string,
             mu.neurons_per_layer_from_string)
        })
        return orig
    
    def save(self, regr):
        print("Can't save a Pipeline containing keras models because they are not pickable")
        print("Saving the configuration instead. Loading will be expensive because it will retrain")
        
        all_params = regr.regressor.steps[1][1].get_params()
        joblib.dump(all_params, self.model_path())

    def load(self):        
        all_params = joblib.load(self.model_path())
        regr = self.make(all_params["model_"])

        print("Fitting with the test dataset")
        X, y = self.load_train_dataset()
        ps.set_all_seeds(all_params["model_"]["seeds"])
        regr.fit(X, y)
        return regr
    
    def predict_test(self, regr, X):
        K.clear_session()
        return self.predict(regr, X)

    def string(self):
        return "DNN"
