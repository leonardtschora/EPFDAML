from epftoolbox.evaluation import MAE, MAPE, MASE, rMAE, RMSE, sMAPE, DM
from epftoolbox.data import read_data
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import os, math
import numpy as np, pandas
import itertools
from matplotlib.colors import ListedColormap
from matplotlib import cm

from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error

def mae(ytrue, ypred): return mean_absolute_error(ytrue, ypred)
def mape(ytrue, ypred): return 100 * mean_absolute_percentage_error(ytrue, ypred)
def smape(ytrue, ypred): return 200 * np.mean(np.abs(ytrue - ypred) / (np.abs(ytrue) + np.abs(ypred)))
def rmse(ytrue, ypred): return math.sqrt(mean_squared_error(ytrue, ypred))
def rmae(ytrue, ypred, ynaive): return mean_absolute_error(ytrue, ypred) / mean_absolute_error(ytrue, ynaive)
def dae(ytrue, ypred):
    try:
        error = mae(ytrue.mean(axis=1), ypred.mean(axis=1))
    except:
        error = mae(ytrue, ypred)
    return error

def data_std(ytrue, ypred): return ytrue.std()


def load_prevs_mw(model_wrapper, dataset="test_recalibrated"):
    if dataset == "test":
        path = model_wrapper.test_prediction_path()
    if dataset == "test_recalibrated":
        path = model_wrapper.test_recalibrated_prediction_path()
    if dataset == "validation":
        path = model_wrapper.val_prediction_path()

    data = pandas.read_csv(path)    
    return data

def get_data(country):
    # Get all forecasts
    forecast = pd.read_csv('https://raw.githubusercontent.com/jeslago/epftoolbox/master/' + 
                           f'forecasts/Forecasts_{country}_DNN_LEAR_ensembles.csv',
                           index_col=0)
    forecast.index = pd.to_datetime(forecast.index)
    
    # Get real prices
    df_train, df_test = read_data(path='.', dataset=country,
                                  begin_test_date=forecast.index[0], 
                                  end_test_date=forecast.index[-1])
    real_price = df_test.loc[:, ['Price']]
    real_price_insample = df_train.loc[:, ['Price']]
    real_price_2D = pd.DataFrame(real_price.values.reshape(-1, 24), 
                                 index=real_price.index[::24], 
                                 columns=['h' + str(hour) for hour in range(24)])
    real_price_insample_2D = pd.DataFrame(real_price_insample.values.reshape(-1, 24), 
                                          index=real_price_insample.index[::24], 
                                          columns=['h' + str(hour) for hour in range(24)])
    val_price = df_train.loc[:, ['Price']]
    val_price_2D = pd.DataFrame(val_price.values.reshape(-1, 24), 
                                index=val_price.index[::24], 
                                columns=['h' + str(hour) for hour in range(24)])
    val_price_2D = val_price_2D.iloc[-362:]

    # Remove the first 7 days of the test set because they can contain data
    # present in the validation or train set.
    real_price_2D = real_price_2D.tail(-7)

    # Remove the first 7 seven days of the train set because they are not used for
    # Forecasting because of the lag!
    real_price_insample_2D = real_price_insample_2D.tail(-7)
    return forecast, real_price_2D, real_price_insample_2D, val_price_2D

def compute_metrics(ytrue, ypred, yinsample,
                    order="default"):
    
    metrics = [MAE(p_pred=ypred, p_real=ytrue),
               MASE(p_pred=ypred, p_real=ytrue, p_real_in=yinsample, m='W'),
               rMAE(p_pred=ypred, p_real=ytrue),
               MAPE(p_pred=ypred, p_real=ytrue) * 100,
               sMAPE(p_pred=ypred, p_real=ytrue) * 100,
               RMSE(p_pred=ypred, p_real=ytrue)]
    if order == "default": return metrics
    default = ["mae", "mase", "rmae", "mape", "smape", "rmse"]
    res = []
    for metric in order:
        for m, value in zip(default, metrics):
            if m == metric: res.append(value)
    return res
    
    
def load_prevs_git(model_name, country="FR"):
    data = pd.read_csv('https://raw.githubusercontent.com/jeslago/epftoolbox/master/' + 
                       f'forecasts/Forecasts_{country}_DNN_LEAR_ensembles.csv',
                       index_col=0)
    data.index = pd.to_datetime(data.index)
    data = data.loc[:, [model_name]]
    data = pd.DataFrame(data.values.reshape(-1, 24), 
                        index=data.index[::24], 
                        columns=['h' + str(hour) for hour in range(24)])    
    return data

def load_prevs(model_wrapper, country="FR", test_dates=None,
               dataset="test_recalibrated", path=os.path.join(
               os.environ["EPFDAML"], "data", "datasets")):
    path = os.path.join(path, f"EPF_{country}")
    data = pd.read_csv(os.path.join(
        path, model_name + "_" + dataset + "_predictions.csv"))
    data.columns = ["h" + dc for dc in data.columns]
    data.index = test_dates
    return data

def all_combis(arr):
    res = [r for r in itertools.combinations(arr, r=1)]
    for i in range(1, len(arr)):
        for r in itertools.combinations(arr, r=i+1):
            res.append(r)
    return res

def load_multiple(model_name, test_dates, dataset):
    data = load_prevs(model_name[0], location="drive", dataset=dataset,
                      test_dates=test_dates)
    if len(model_name) > 1:
        for mn in model_name[1:]:
            data += load_prevs(mn, location="drive", dataset=dataset,
                          test_dates=test_dates)
        data /= len(model_name)

    return data

def cmap():
    red = np.concatenate([np.linspace(0, 1, 50), np.linspace(1, 0.5, 50)[1:], [0]])
    green = np.concatenate([np.linspace(0.5, 1, 50), np.zeros(50)])
    blue = np.zeros(100)
    rgb_color_map = np.concatenate([red.reshape(-1, 1), green.reshape(-1, 1), 
                                    blue.reshape(-1, 1)], axis=1)
    rgb_color_map = mpl.colors.ListedColormap(rgb_color_map)
    return rgb_color_map

def cmap_2():
    red = np.concatenate([np.linspace(0, 1, 50), [0]])
    green = np.concatenate([np.linspace(0.5, 1, 50), [0]])
    blue = np.zeros(51)
    rgb_color_map = np.concatenate([red.reshape(-1, 1), green.reshape(-1, 1), 
                                    blue.reshape(-1, 1)], axis=1)
    rgb_color_map = mpl.colors.ListedColormap(rgb_color_map)
    return rgb_color_map

def cmap_scaled_values():
    greens = cm.get_cmap('Greens', 1000)
    reds = cm.get_cmap('Reds', 1000)
    newcolors = np.concatenate((greens(np.linspace(0.75, 0, 250)),
                                reds(np.linspace(0, 0.8, 1000))))
    newcolors[0, :] = greens(1000)
    my_cmap = ListedColormap(newcolors)
    return my_cmap

def cmap_shap_values():
    reds = cm.get_cmap('Reds', 1000) 
    greens = cm.get_cmap('Greens', 1000)   
    newcolors = np.concatenate((greens(np.linspace(0, 1, 200)),
                                reds(np.linspace(0.5, 1, 800))))
    newcolors[0, :] = np.array([0, 0, 0, 1])
    my_cmap = ListedColormap(newcolors)
    return my_cmap

def cmap_diff_values():
    greens = cm.get_cmap('Greens', 1000)
    reds = cm.get_cmap('Reds', 1000)
    newcolors = np.concatenate((reds(np.linspace(1, 0.5, 250)),
                                greens(np.linspace(0, 1, 1000))))
    my_cmap = ListedColormap(newcolors)
    return my_cmap

def cmap_diff_values_2():
    greens = cm.get_cmap('Blues', 1000)
    reds = cm.get_cmap('Oranges', 1000)
    newcolors = np.concatenate((reds(np.linspace(1, 0, 1000)),
                                greens(np.linspace(0, 1, 1000))))
    my_cmap = ListedColormap(newcolors)
    return my_cmap
