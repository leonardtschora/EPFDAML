from work.models.model_wrapper import *
from work.models.NeuralNetworks.CNN import LeNet
from work.models.NeuralNetWrapper import NeuralNetWrapper

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import BaggingRegressor
from tensorflow.keras import initializers
from tensorflow.keras import regularizers
from tensorflow.keras.models import load_model
import work.parallel_scikit as ps 

from work.sampling.samplers import CNN_space

class LeNetWrapper(NeuralNetWrapper):
    def __init__(self, prefix, dataset_name, label, date_cols=[], columns=[],
                 spliter=None, W=None, H=24):
        NeuralNetWrapper.__init__(self, prefix, dataset_name, label,spliter=spliter)
        self.date_cols = date_cols 
        self.columns = columns        
        # Floor the result, so we have to increase W if we add the gaz prices
        if W is None: W = int((len(self.columns) - len(self.date_cols)) / 24)
        if "FR_ShiftedGazPrice" in columns: W += 1
        self.W = W        
        self.H = H
        
    def params(self):
        orig = NeuralNetWrapper.params(self)
        orig.update({"conv_activation" : "relu",
                     "filter_size" : ((6, 16, ), ),
                     "dilation_rate" : (((1, 1), (1, 1), ), ),
                     
                     "kernel_size" : (((5, 5), (5, 5), ), ), 
                     "pool_size" : ((2, 2),  ),
                     "strides" : ((2, 2), ),
        })
        return orig

    def map_dict(self):
        orig = NeuralNetWrapper.map_dict(self)
        orig.update({"structure" :
                     {
                         "filter_size" : (mu.filter_size_to_string,
                                          mu.filter_size_from_string),
                         "dilation_rate" : (mu.dilation_rate_to_string,
                                            mu.dilation_rate_from_string),
                         "kernel_size" : (mu.dilation_rate_to_string,
                                          mu.dilation_rate_from_string),
                         "pool_size" : (mu.dilation_rate_to_string,
                                        mu.dilation_rate_from_string),
                         "strides" : (mu.neurons_per_layer_to_string,
                                      mu.neurons_per_layer_from_string),
                     }                    
        })
        return orig

    def make(self, ptemp):
        ptemp_ = copy.deepcopy(ptemp)
        if "seeds" in ptemp_:
            del ptemp_["seeds"]

        scaler = self.get_scaler(ptemp_)
        transformer = self.get_transformer(ptemp_)
        
        model = LeNet("test", ptemp_, self.W, self.H)
        pipe = make_pipeline(scaler, model)
        regr = TransformedTargetRegressor(pipe, transformer=transformer)

        return regr

    def load_dataset(self, path):
        """
        Sort the columns so the dataset is correclty reshaped
        """
        dataset = pandas.read_csv(path)
        labels = dataset[np.array(self.label)]
        dataset.drop(columns=self.date_cols + self.label + ["period_start_date"],
                     inplace=True)
        
        names = [c for c in self.columns if c not in self.date_cols]
        
        # Duplicate the gaz prices
        col = "FR_ShiftedGazPrice"
        if col in names:
            names = names[np.where(np.array(names) == col)[0][0]+1:]
            for i in range(self.H):
                name = col + f"_{i}"
                names.append(name)
                dataset[name] = dataset[col]
        
        # Sort the names
        dataset = dataset[names]            
        X = dataset.values 
        y = labels.values        
        return X, y
    
    def predict_val(self, regr, X, oob=False):
        if oob:
            print("Can't access the oob prediction!")
        else:
            return self.predict(regr, X)
    
    def eval_val(self, regr, X, y, oob=False):
        """
        Use the out of sample validation loss to provide an estimate of the 
        generalization error. 
        """
        if not oob:
            return self.eval(regr, X, y)
        else:
            scaled_loss = regr.regressor_.steps[1][1].callbacks[0].val_losses[-1]
            return scaled_loss

    def get_search_space(self, country=None, version=None, n=None,
                         fast=False, stop_after=-1):
        return CNN_space(n, self.W, self.H, country, fast=fast,
                         stop_after=stop_after)

    def string(self):
        return "CNN"    
