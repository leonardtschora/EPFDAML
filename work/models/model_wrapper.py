import os, copy, joblib, pandas
import work.model_utils as mu
import numpy as np
import work.parallel_scikit as ps
import time, shap
from sklearn.metrics import mean_absolute_error
from work.analysis.evaluate import mae, smape, mape, rmse, rmae, dae, cmap
from epftoolbox.evaluation import MAE, MAPE, MASE, rMAE, RMSE, sMAPE
from work.models.DataScaler import DataScaler
from sklearn.model_selection import PredefinedSplit
import work.parallel_scikit as ps

class ModelWrapper(object):
    """
    Model wrapper around ensemble of models. Facilitates the workflows around
    models in general and enables parallel grid searches around models.
    Enables using the oob's predictions for evaluating a model's quality during grid
    searches.
    """
    def __init__(self, prefix, dataset_name, label, spliter=None):
        self.prefix = prefix
        self.dataset_name = dataset_name
        self.label = label

        # Spliter to give to all components of the model
        self.spliter = spliter

    def string(self):
        return "ModelWrapper"

    def shuffle_train(self):
        return True
        
    def save_name(self):
        return mu.save_name(self.prefix, self.dataset_name)

    def save_path(self):
        return mu.save_path(self.prefix, self.dataset_name)
    
    def folder(self):
        folder = mu.folder(self.dataset_name)
        if not os.path.exists(folder): os.mkdir(folder)
        return folder

    def results_path(self):
        return self.save_path() + "_results.csv" 
    
    def train_dataset_path(self):
        return mu.train_dataset_path(self.dataset_name)

    def test_dataset_path(self):
        return mu.test_dataset_path(self.dataset_name)

    def all_dataset_path(self):
        return mu.all_dataset_path(self.dataset_name)

    def extra_dataset_path(self):
        return mu.extra_dataset_path(self.dataset_name)        

    def figure_folder(self):
        return mu.figure_folder(self.dataset_name)

    def model_path(self):
        return mu.model_path(self.prefix, self.dataset_name)

    def train_prediction_path(self):
        return mu.train_prediction_path(self.prefix, self.dataset_name)

    def test_prediction_path(self):
        return mu.test_prediction_path(self.prefix, self.dataset_name)

    def val_prediction_path(self):
        return mu.val_prediction_path(self.prefix, self.dataset_name)    

    def test_recalibrated_prediction_path(self):
        return mu.test_recalibrated_prediction_path(self.prefix, self.dataset_name)
    
    def extra_prediction_path(self):
        return mu.extra_prediction_path(self.prefix, self.dataset_name)

    def all_prediction_path(self):
        return mu.all_prediction_path(self.prefix, self.dataset_name)

    def test_shape_path(self):
        return mu.test_shape_path(self.prefix, self.dataset_name)

    def test_recalibrated_shape_path(self):
        return mu.test_recalibrated_shape_path(self.prefix, self.dataset_name)

    def _params(self, ptemp):
        p = self.params()
        p.update(ptemp)
        return p

    def get_scaler(self, ptemp):
        try:
            scaler = DataScaler(ptemp["scaler"], spliter=self.spliter)
        except:
            scaler = DataScaler("", spliter=self.spliter)
            
        try: del ptemp["scaler"]
        except: pass
        
        return scaler

    def get_transformer(self, ptemp):
        try:
            transformer = DataScaler(ptemp["transformer"], spliter=self.spliter)
        except:
            transformer = DataScaler("", spliter=self.spliter)
            
        try: del ptemp["transformer"]
        except: pass
        
        return transformer

    def get_search_space(self, version=None, n=None, stop_after=-1):
        pass        
    
    def params(self):
        return {}

    def make(self, ptemp):
        pass

    def predict(self, regr, X):
        return regr.predict(X)
    
    def eval(self, regr, X, y): 
        yhat = self.predict(regr, X)
        return mean_absolute_error(y, yhat)       
    
    def predict_val(self, regr, X, oob=False):
        return self.predict(regr, X)
    
    def eval_val(self, regr, X, y, oob=False):
        return self.eval(regr, X, y)

    def predict_test(self, regr, X):
        return self.predict(regr, X)
    
    def eval_test(self, regr, X, y):
        return self.eval(regr, X, y)        
        
    def save(self, model):
        joblib.dump(model, self.model_path())

    def load(self):
        return joblib.load(self.model_path())

    def load_dataset(self, path):
        return mu.load_dataset(path, self.label)
    
    def load_train_dataset(self):
        return self.load_dataset(self.train_dataset_path())
    
    def load_test_dataset(self):
        return self.load_dataset(self.test_dataset_path())

    def load_prediction_dataset(self):
        dataset = pandas.read_csv(self.test_dataset_path())
        dataset.drop(columns=["period_start_date"], inplace=True)
        X = dataset.values
        return X

    def load_base_results(self):
        p1 = os.path.split(mu.folder(self.dataset_name))[0]
        p2 = os.path.split(self.dataset_name)[0]
        p3 = self.prefix
        path = os.path.join(p1, f"{p3}_{p2}_results.csv")
        return self.load_results(path=path)

    def load_recalibration_dataset(self):
        """
        The extra dataset is constitued of the unused data = the week of data 
        removed from the test set. IN recalibration mode, this data is integrated 
        into the test set.
        """
        Xtrain, ytrain = self.load_dataset(self.train_dataset_path())
        Xextra, yextra = self.load_dataset(self.extra_dataset_path())
        X = np.concatenate((Xtrain, Xextra))
        Y = np.concatenate((ytrain, yextra))
        return X, Y
        
    def map_dict(self):
        return {"spliter" : (mu.spliter_to_string, mu.spliter_from_string)}

    def load_results(self, path=None):
        if path is None: path = self.results_path()
        df = pandas.read_csv(path)
        
        map_dict = self.map_dict()
        for k in map_dict.keys():
            # Handle nested params
            if type(map_dict[k]) == dict:
                for k2 in map_dict[k].keys():
                    _, f = map_dict[k][k2]
                    df[k2] = [f(i) for i in df[k2]]
            else:
                _, f = map_dict[k]
                if k in df.keys(): df[k] = [f(i) for i in df[k]]
        return df        

    def best_params(self, df, for_recalibration=False):
        best_row = df.maes.argmin()
        best_params = df.loc[best_row].to_dict()
        
        params = self.params()
        params.update(best_params)

        if for_recalibration:
            if "stop_after" in params.keys():
                params["stop_after"] = -1
            params.pop("maes")
            params.pop("times")
            
        return params

    def save_preds(self, regr, trades_col):
        X, y = self.load_train_dataset()        
        Xt, yt = self.load_test_dataset()
        self.save_train_preds(regr, X, trades_col)
        self.save_test_preds(regr, Xt, trades_col)       
        
    def save_train_preds(self, regr, X, trades_col):
        train_dataset = pandas.read_csv(self.train_dataset_path())
        train_prevs = pandas.DataFrame({"date_col" : train_dataset["date_col"].values,
                                        "predicted_prices" : self.predict(regr, X),
                                        "real_prices" : train_dataset[self.label],
                                        "trades" : train_dataset[trades_col]})
        train_prevs.to_csv(self.train_prediction_path(), index=False)

    def save_test_preds(self, regr, Xt, trades_col):
        test_dataset = pandas.read_csv(self.test_dataset_path())
        test_prevs = pandas.DataFrame({"date_col" : test_dataset["date_col"].values,
                                       "predicted_prices" : regr.predict(Xt),
                                       "real_prices" : test_dataset[self.label],
                                       "trades" : test_dataset[trades_col]})
        test_prevs.to_csv(self.test_prediction_path(), index=False)     

    def test_and_save(self, best_params, trades_col, trades_dataset,
                      cv=None, same_val=False, oob=False):
        X, y = self.load_train_dataset()        
        Xt, yt = self.load_test_dataset()

        for k in self.map_dict().keys():
            v = self.map_dict()[k]
            best_params[k] = v[1](best_params[k]) 
        
        regr = self.make(self._params(best_params))
        regr.fit(X, y)               

        self.save(regr)

        print("Best model saved at " + self.model_path())
        print("TEST SET MAE = " + str(mu.mae_epsilon(yt, regr.predict(Xt))))
        print("TEST SET SMAPE = " + str(mu.smape_epsilon(yt, regr.predict(Xt))))
        
        ##### Trade metrics
        test_trades_path = mu.test_dataset_path(trades_dataset)
        test_trades = pandas.read_csv(test_trades_path)[trades_col].values
        print("TEST SET GAIN = " + str(mu.gain(yt, regr.predict(Xt), test_trades)))
        print("TEST SET DIRECTION = " + str(mu.direction(yt, regr.predict(Xt), test_trades)))
        return regr

    def set_jobs(self, regr, njobs):
        pass

    def test_and_save_epf(self, best_params, seed=None, njobs=1, save=True,
                          validation_mode="external", external_spliter=None):
        regr = self.make(self._params(best_params))
        self.set_jobs(regr, njobs)
        oob = validation_mode == "oob"
        
        # Recompute validation error
        X, y = self.load_train_dataset()        
        X, y, Xv, yv = ps.outer_validation(validation_mode, external_spliter, X, y)
        Xv, yv = ps.inner_validation(validation_mode, X, y, Xv, yv, self)
            
        if seed is not None: ps.set_all_seeds(seed)
        regr.fit(X, y)
        ypred = self.predict_val(regr, Xv, oob=oob)
        print("(RECOMPUTED) VAL SET MAE = " + str(round(mae(yv, ypred), ndigits=2)))
        print("(RECOMPUTED) VAL SET SMAPE = " + str(round(smape(yv, ypred), ndigits=2)))
        print("(RECOMPUTED) VAL SET DAE = " + str(round(dae(yv, ypred), ndigits=2) )) 

        val_prevs = pandas.DataFrame(ypred)
        if save: val_prevs.to_csv(self.val_prediction_path(), index=False)

        # Compute test set error
        X, y = self.load_train_dataset()
        Xt, yt = self.load_test_dataset()
        
        if seed is not None: ps.set_all_seeds(seed)
        regr.fit(X, y)          
        
        ypred = self.predict_test(regr, Xt)
        test_prevs = pandas.DataFrame(ypred)
        if save: test_prevs.to_csv(self.test_prediction_path(), index=False)
        print("TEST SET MAE = " + str(round(mae(yt, ypred), ndigits=2)))
        print("TEST SET SMAPE = " + str(round(smape(yt, ypred), ndigits=2)))
        print("TEST SET DAE = " + str(round(dae(yt, ypred), ndigits=2)))

        if save: self.save(regr)
        print("Best model saved at " + self.model_path())            
        return regr 

    def recalibrate_epf(self, regr=None, best_params={}, ncpus=1, seed=None,
                        start=None, stop=None, calibration_window=None,
                        save=True, n_shap=0):
        if self.spliter is not None and self.spliter.shuffle:
            print("Shuffling the validation set")

        # Load data
        Xtrain, ytrain = self.load_recalibration_dataset()
        Xt, yt = self.load_test_dataset()

        # Handle start and stop if unspecified
        if start is None: start = 0
        if stop is None: stop = yt.shape[0]

        # Select the data between start and stop
        Xtrain = np.concatenate((Xtrain, Xt[:start]))
        ytrain = np.concatenate((ytrain, yt[:start]))            
        Xt = Xt[start:stop]
        yt = yt[start:stop]
        ntest = yt.shape[0]

        # Paralellize the computations
        predictions = np.zeros_like(yt)
        times = np.zeros(ntest)
        
        try:
            n_labels = yt.shape[1]
        except:
            n_labels = 1
        shaps = np.zeros((n_labels, yt.shape[0], Xt.shape[1]))
        print(f"RECALIBRATING {stop -start} DAYS ON {calibration_window} SAMPLES")
        ps.recalibrate_parallel(predictions, self, best_params, Xtrain, ytrain,
                                Xt, yt, seed, times, ncpus, ntest, shaps=shaps,
                                calibration_window=calibration_window,
                                n_shap=n_shap)

        # Compute metrics
        print("RECALIBRATED TEST SET MAE = ",
              str(round(mae(yt, predictions), ndigits=2)))
        print("RECALIBRATED TEST SET SMAPE = ",
              str(round(smape(yt, predictions), ndigits=2)))
        print("RECALIBRATED TEST SET DAE = ",
              str(round(dae(yt, predictions), ndigits=2)))
        total_time = np.sum(times)
        print(f"RECALIBRATED TEST TIME = " + str(total_time))

        # Save recalibrated predictions
        test_prevs = pandas.DataFrame(predictions)
        if save: test_prevs.to_csv(
                self.test_recalibrated_prediction_path(), index=False)
        if n_shap > 0: np.save(self.test_recalibrated_shape_path(), shaps)
        return total_time

        
        
