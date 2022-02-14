from work.models.model_wrapper import *
from work.models.NeuralNetworks.DNN import DNN
from work.models.NeuralNetWrapper import NeuralNetWrapper
import tensorflow.keras.backend as K
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import BaggingRegressor
from tensorflow.keras import initializers
from tensorflow.keras import regularizers
from tensorflow.keras.models import load_model
import work.parallel_scikit as ps 

from work.sampling.samplers import MLP_space

class MLPWrapper(NeuralNetWrapper):
    def __init__(self, prefix, dataset_name, label, spliter=None):
        NeuralNetWrapper.__init__(self, prefix, dataset_name, label,spliter=spliter)
        
    def params(self):
        orig = NeuralNetWrapper.params(self)
        orig.update()
        return orig

    def map_dict(self):
        orig = NeuralNetWrapper.map_dict(self)
        orig.update({})
        return orig       

    def make(self, ptemp):
        ptemp_ = copy.deepcopy(ptemp)
        if "seeds" in ptemp_:
            del ptemp_["seeds"]

        scaler = self.get_scaler(ptemp_)
        transformer = self.get_transformer(ptemp_)
            
        model = DNN("test", ptemp_)
        pipe = make_pipeline(scaler, model)
        regr = TransformedTargetRegressor(pipe, transformer=transformer)

        return regr

    def predict(self, regr, X):
        predictions = regr.predict(X)
        return predictions               
    
    def predict_val(self, regr, X, oob=False):
        if oob: print("Can't access the oob prediction!")
        else: return self.predict(regr, X)
        
    def eval_val(self, regr, X, y, oob=False):
        """
        Use the out of sample validation loss to provide an estimate of the 
        generalization error. 
        """
        if not oob:
            yhat = self.predict_val(regr, X, oob=oob)
            return mean_absolute_error(y, yhat)
        else:
            scaled_loss = regr.regressor_.steps[1][1].callbacks[0].val_losses[-1]
            return scaled_loss

    def get_search_space(self, country, version=None,  n=None, fast=False,
                         stop_after=-1):
        return MLP_space(n, country, fast=fast, stop_after=stop_after)    

class MLP_ENSWrapper(NeuralNetWrapper):
    def __init__(self, prefix, dataset_name, label, ensemble_size=30, max_samples=1.0):
        NeuralNetWrapper.__init__(self, prefix, dataset_name, label)
        self.ensemble_size=ensemble_size
        self.max_samples = max_samples

        
    def params(self):
        orig = NeuralNetWrapper.params(self)
        orig.update({})
        return orig

    def map_dict(self):
        orig = NeuralNetWrapper.map_dict(self)
        orig.update({})
        return orig      
        

    def make(self, ptemp):
        scaler = self.get_scaler(ptemp)
        transformer = self.get_transformer(ptemp)
        
        model = DNN("test", ptemp)
        pipe = make_pipeline(scaler, model)
        regr = TransformedTargetRegressor(pipe, transformer=transformer)
        
        ensemble = BaggingRegressor(base_estimator=regr,
                                    n_estimators=self.ensemble_size,
                                    max_samples=self.max_samples,
                                    oob_score=False,
                                    n_jobs=1)

        ptemp["scaler"] = scaler
        ptemp["transformer"] = transformer        
        return ensemble 

    def predict(self, regr, X):
        return regr.predict(X)
        
    def predict_val(self, regr, X, oob=False):
        if oob: return regr.oob_prediction_
        else: return self.predict(regr, X)
        
    def eval_val(self, regr, X, y, oob=False):        
        yhat = self.predict_val(regr, X, oob=oob)
        return mean_absolute_error(y, yhat)

    def save(self, regr):
        print("Can't save an ensemble of keras models because they are not pickable")
        print("Saving the configuration instead. Loading will be expensive because it will retrain")
        
        all_params = regr.base_estimator_.regressor.steps[1][1].get_params()
        joblib.dump(all_params, self.model_path())        

    def set_jobs(self, regr, njobs):
        regr.n_jobs=njobs

    def string(self):
        return "MLP"    
