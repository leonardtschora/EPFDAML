from work.models.model_wrapper import *
import work.model_utils as mu

class Feature(ModelWrapper):
    def __init__(self, prefix, dataset_name, label, trade_col_idx):
        ModelWrapper.__init__(self, prefix, dataset_name, label)
        self.trade_col_idx = trade_col_idx

    def params(self):
        return {}
    
    def make(self, ptemp):
        return None

    def predict(self, regr, X):
        return X[:, self.trade_col_idx]
    
    def eval(self, regr, X, y):
        return mean_absolute_error(y, self.predict(regr, X))
        
