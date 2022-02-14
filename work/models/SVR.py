from work.models.model_wrapper import *
from sklearn import svm
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import TransformedTargetRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.multioutput import MultiOutputRegressor, RegressorChain

from work.sampling.samplers import SVR_space
import copy

class SVR(ModelWrapper):
    def __init__(self, prefix, dataset_name, label, spliter=None):
        ModelWrapper.__init__(self, prefix, dataset_name, label, spliter=spliter)
        
    def params(self):
        return {"kernel":"rbf",
                "tol":0.0001,
                "C":1.0,
                "epsilon":0.01,
                "shrinking":True,
                "max_iter":500000,
                "cache_size":200,
                "scaler":"Standard",
                "transformer":"Standard"}

    def make(self, ptemp):
        ptemp_ = copy.deepcopy(ptemp)
        if "seeds" in ptemp_:
            del ptemp_["seeds"]        
        
        scaler = self.get_scaler(ptemp_)
        transformer = self.get_transformer(ptemp_)
        
        model = svm.SVR(**ptemp_)
        pipe = make_pipeline(scaler, model)
        regr = TransformedTargetRegressor(pipe, transformer=transformer)
        
        return regr  

    def get_search_space(self, country=None, version=None, n=None,
                         fast=False, stop_after=None):
        return SVR_space(fast=fast)
 
    def string(self):
        return "SVR"   


class MultiSVR(SVR):
    def __init__(self, prefix, dataset_name, label, spliter=None):
        SVR.__init__(self, prefix, dataset_name, label, spliter=spliter)
        
    def params(self):
        return {"kernel":"rbf",
                "tol":0.0001,
                "C":1.0,
                "epsilon":0.01,
                "shrinking":True,
                "max_iter":500000,
                "cache_size":200,
                "scaler":"Standard",
                "transformer":"Standard"}

    def make(self, ptemp):
        ptemp_ = copy.deepcopy(ptemp)
        if "seeds" in ptemp_:
            del ptemp_["seeds"]        
        
        scaler = self.get_scaler(ptemp_)
        transformer = self.get_transformer(ptemp_)
        
        model = MultiOutputRegressor(svm.SVR(**ptemp_))
        pipe = make_pipeline(scaler, model)
        regr = TransformedTargetRegressor(pipe, transformer=transformer)
        
        return regr    

    def string(self):
        return "SVR Multi"
    
class ChainSVR(SVR):
    def __init__(self, prefix, dataset_name, label, spliter=None):
        SVR.__init__(self, prefix, dataset_name, label, spliter=spliter)
        
    def params(self):
        return {"kernel":"rbf",
                "tol":0.0001,
                "C":1.0,
                "epsilon":0.01,
                "shrinking":True,
                "max_iter":500000,
                "cache_size":200,
                "scaler":"Standard",
                "transformer":"Standard"}

    def make(self, ptemp):
        ptemp_ = copy.deepcopy(ptemp)
        if "seeds" in ptemp_:
            del ptemp_["seeds"]
            
        scaler = self.get_scaler(ptemp_)
        transformer = self.get_transformer(ptemp_)
        
        model = RegressorChain(svm.SVR(**ptemp_))
        pipe = make_pipeline(scaler, model)
        regr = TransformedTargetRegressor(pipe, transformer=transformer)

        return regr    
    
    def string(self):
        return "SVR Chain"    


class MetaSVR(ModelWrapper):
    def __init__(self, prefix, dataset_name, label, spliter=None):
        ModelWrapper.__init__(self, prefix, dataset_name, label, spliter=spliter)
        
    def params(self):
        return {"n_combis" : 4000,
                "n_best" : 20,}

    def make(self, ptemp):
        ptemp_ = copy.deepcopy(ptemp)
        if "seeds" in ptemp_:
            del ptemp_["seeds"]        
        
        scaler = self.get_scaler(ptemp_)
        transformer = self.get_transformer(ptemp_)
        
        model = svm.SVR(**ptemp_)
        pipe = make_pipeline(scaler, model)
        regr = TransformedTargetRegressor(pipe, transformer=transformer)
        
        return regr  

    def get_search_space(self, country=None, version=None, n=None,
                         fast=False, stop_after=None):
        return SVR_space(fast=fast)
 
    def string(self):
        return "SVR"       
