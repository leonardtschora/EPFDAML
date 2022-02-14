import pandas, os
import work.parallel_scikit as ps
from epftoolbox.models import hyperparameter_optimizer, format_best_trial
from epftoolbox.models._dnn import _build_and_split_XYs
from epftoolbox.data import read_data, scaling
from epftoolbox.evaluation import MAE, sMAPE
from hyperopt import Trials
import pandas as pd
import pickle
import numpy as np
from epftoolbox.models import DNNModel
from epftoolbox.models import evaluate_dnn_in_test_dataset

def get_path(country):
    return os.path.join(os.environ["EPFDAML"], "data", "datasets", "EPF_" + country)

def optim_once(iter_, max_evals, shuffle_train, seed):
    ps.set_all_seeds(seed)    
    country, layer = iter_
    print(seed, country, layer)
    data_path = os.path.join(os.environ["EPFDAML"], "data", "datasets", "EPF_" + country)  
    hyperparameter_optimizer(path_datasets_folder=data_path, 
                             path_hyperparameters_folder=data_path, 
                             new_hyperopt=True, max_evals=max_evals, nlayers=layer,
                             dataset=country, years_test=2, calibration_window=4, 
                             shuffle_train=shuffle_train, data_augmentation=False,
                             experiment_id="LAGO_RESULTS_" + str(seed),
                             begin_test_date=None, end_test_date=None)

def load_results(all_combis_seeds):
    df = pandas.DataFrame(
        index=range(len(all_combis_seeds)),
        columns=['seed', 'country', 'nlayer','loss', 'MAE Val', 'MAE Test',
                 'sMAPE Val', 'sMAPE Test', 'status'])
    for i, (seed, country, layer) in enumerate(all_combis_seeds):
        path = os.path.join(os.environ["EPFDAML"], "data", "datasets", "EPF_" + country)
        filename = f"DNN_hyperparameters_nl{layer}_dat{country}_YT2_CW4_LAGO_RESULTS_{seed}"
        with open(os.path.join(path, filename), "rb") as f:
            res = pickle.load(f)

        results = res.best_trial['result']
        results.update({"seed" : seed, "country" : country, "nlayer" : layer})
        df.loc[i] = results

    df.drop(columns="status", inplace=True)
    return df


def predict_config(res, path, best_nlayer, country, shuffle_train):
    hyperparameters = format_best_trial(res.best_trial)
    dfTrain, dfTest = read_data(path, dataset=country)
    dfTrain_cw = dfTrain.loc[dfTrain.index[-1] - pd.Timedelta(weeks=52) * 4 + pd.Timedelta(hours=1):]
    Xtrain, Ytrain, Xval, Yval, Xtest, Ytest, indexTest =  _build_and_split_XYs(
        dfTrain=dfTrain_cw, dfTest=dfTest, features=hyperparameters, 
        shuffle_train=shuffle_train, hyperoptimization=True,
        data_augmentation=False, n_exogenous_inputs=2)
    
    # If required, datasets are scaled
    if hyperparameters['scaleX'] in ['Norm', 'Norm1', 'Std', 'Median', 'Invariant']:
        [Xtrain, Xval, Xtest], _ = scaling([Xtrain, Xval, Xtest], hyperparameters['scaleX'])

    if hyperparameters['scaleY'] in ['Norm', 'Norm1', 'Std', 'Median', 'Invariant']:
        [Ytrain, Yval], scaler = scaling([Ytrain, Yval], hyperparameters['scaleY'])
    else:
        scaler = None
    
    neurons = [int(hyperparameters['neurons' + str(k)]) for k in range(1, best_nlayer + 1)
               if int(hyperparameters['neurons' + str(k)]) >= 50]        

    ps.set_all_seeds(int(hyperparameters['seed']), env=False)

    # Initialize model
    forecaster = DNNModel(neurons=neurons, n_features=Xtrain.shape[-1], 
                          dropout=hyperparameters['dropout'],
                          batch_normalization=hyperparameters['batch_normalization'], 
                          lr=hyperparameters['lr'], verbose=False,
                          optimizer='adam', activation=hyperparameters['activation'],
                          epochs_early_stopping=20, scaler=scaler, loss='mae',
                          regularization=hyperparameters['reg'], 
                          lambda_reg=hyperparameters['lambdal1'],
                          initializer=hyperparameters['init'])

    forecaster.fit(Xtrain, Ytrain, Xval, Yval)
    Yp_val = forecaster.predict(Xval).squeeze()
    if hyperparameters['scaleY'] in ['Norm', 'Norm1', 'Std', 'Median', 'Invariant']:
        Yval = scaler.inverse_transform(Yval)
        Yp_val = scaler.inverse_transform(Yp_val)

    # If required, datasets are normalized
    Yp_test = forecaster.predict(Xtest).squeeze()
    if hyperparameters['scaleY'] in ['Norm', 'Norm1', 'Std', 'Median', 'Invariant']:
        Yp_test = scaler.inverse_transform(Yp_test).squeeze()

    print("(RECOMPUTED) VAL SET MAE = " + str(MAE(Yval, Yp_val)))
    print("(RECOMPUTED) TEST SET MAE = " + str(MAE(Ytest, Yp_test)))
    print("(RECOMPUTED) TEST SET SMAPE = " + str(100 * sMAPE(Ytest, Yp_test)))

    return Yp_val, Yp_test

def load_hp_file(country, best_nlayer, seed):
    path = get_path(country)
    best_filename = f"DNN_hyperparameters_nl{best_nlayer}_dat{country}_YT2_CW4_LAGO_RESULTS_{seed}"
    with open(os.path.join(path, best_filename), "rb") as f:
        res = pickle.load(f)
    return res, path

def test_best_config(df, country, seed, shuffle_train):
    losses = df.loc[np.logical_and(df.seed == seed, df.country == country), :]
    indmin = losses.loss.values.argmin()
    best_nlayer = losses.iloc[indmin].nlayer
    res, path = load_hp_file(country, best_nlayer, seed)
        
    Yp_val, Yp_test = predict_config(res, path, best_nlayer, country, shuffle_train)
    val_pred_file = f"DNN_{seed}_val_predictions.csv"
    test_pred_file = f"DNN_{seed}_test_predictions.csv"
    pandas.DataFrame(Yp_val).to_csv(os.path.join(path, val_pred_file), index=False)
    pandas.DataFrame(Yp_test).to_csv(os.path.join(path, test_pred_file), index=False)
    
def recalibrate(seed, country, df, shuffle_train):
    # Need to retrieve the best number of layers
    losses = df.loc[np.logical_and(df.seed == seed, df.country == country), :]
    indmin = losses.loss.values.argmin()
    best_nlayer = losses.iloc[indmin].nlayer
    
    path = get_path(country)
    recalibrated_predictions =  evaluate_dnn_in_test_dataset(
        f"LAGO_RESULTS_{seed}", path_hyperparameter_folder=path, path_datasets_folder=path,
        shuffle_train=shuffle_train, path_recalibration_folder=path, nlayers=best_nlayer,
        dataset=country, years_test=2, data_augmentation=False, calibration_window=4,
        new_recalibration=True, begin_test_date=None, end_test_date=None)

    # Remove the first 7 predictions because they are not part of the test set
    recalibrated_predictions = recalibrated_predictions.tail(-7)
    
    test_recal_pred_file = f"DNN_{seed}_test_recalibrated_predictions.csv"
    recalibrated_predictions.to_csv(os.path.join(path, test_recal_pred_file), index=False)
    
