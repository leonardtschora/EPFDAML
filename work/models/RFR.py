from work.models.model_wrapper import *
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import TransformedTargetRegressor
from sklearn.metrics import mean_absolute_error

from work.sampling.samplers import RFR_space

class RFR(ModelWrapper):
    def __init__(self, prefix, dataset_name, label, spliter=None):
        ModelWrapper.__init__(self, prefix, dataset_name, label, spliter=spliter)
        
    def params(self):
        return {"n_estimators":100,
                "criterion":'mse',
                "max_depth":None,
                "min_samples_split":2,
                "min_samples_leaf":1,
                "min_weight_fraction_leaf":0.0,
                "max_features":'auto',
                "max_leaf_nodes":None,
                "min_impurity_decrease":0.0,
                "min_impurity_split":None,
                "bootstrap":True,
                "oob_score":True,
                "n_jobs":None,
                "random_state":90125,
                "verbose":0,
                "warm_start":False,
                "ccp_alpha":0.0,
                "max_samples":0.7,
                "scaler":"",
                "transformer":"Standard"}

    def make(self, ptemp):
        ptemp_ = copy.deepcopy(ptemp)
        if "seeds" in ptemp_:
            del ptemp_["seeds"]
            
        scaler = self.get_scaler(ptemp_)
        transformer = self.get_transformer(ptemp_)        
        
        model = RandomForestRegressor(**ptemp_)
        pipe = make_pipeline(scaler, model)
        regr = TransformedTargetRegressor(pipe, transformer=transformer)

        return regr

    def predict_val(self, regr, X, oob=False):
        if oob:
            return regr.transformer_.inverse_transform(
                regr.regressor_.steps[1][1].oob_prediction_)
        else:
            return regr.predict(X)

    def eval_val(self, regr, X, y, oob=False):
        yhat = self.predict_val(regr, X, oob=oob)
        return mean_absolute_error(y, yhat)

    def get_search_space(self, country=None, version=None, n=None,
                         fast=False, stop_after=-1):
        return RFR_space(fast=fast)
    
    def string(self):
        return "RF"
