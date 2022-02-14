from work.models.model_wrapper import *
from sklearn.linear_model import LinearRegression
import work.model_utils as mu
import work.parallel_scikit as ps

class LinearWrapper(ModelWrapper):
    def __init__(self, prefix, dataset_name, label):
        ModelWrapper.__init__(self, prefix, dataset_name, label)

    def params(self):
        return {}
    
    def make(self, ptemp):
        return LinearRegression()
