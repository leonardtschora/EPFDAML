import pandas, copy, numpy as np, os
from epftoolbox.data import read_data
from epftoolbox.models import LEAR, evaluate_lear_in_test_dataset

def get_path(country):
    return os.path.join(os.environ["EPFDAML"], "data", "datasets", "EPF_" + country)

def predict_val_test(calibration_window, country, nval):
    model = LEAR(calibration_window=calibration_window)    
    path = get_path(country)
    df_train, df_test = read_data(path, dataset=country)

    Xtrain_, Ytrain_, Xtest = model._build_and_split_XYs(df_train, df_test=df_test)
    Xtrain, Xval = Xtrain_[:-nval], Xtrain_[-nval:]
    Ytrain, Yval = Ytrain_[:-nval], Ytrain_[-nval:]

    # Validation predictions
    # The "fit" code is only available in the recalibrate function
    # Moreover, the calibration window is handled in the "recalibrate_and_forecast_next_day"
    # function so we have to perform it here  
    model.recalibrate(copy.deepcopy(Xtrain[-calibration_window:]),
                      copy.deepcopy(Ytrain[-calibration_window:]))

    # Somehow the proposed lear model can't make predictions for more than 1 day
    Yp_val = np.zeros_like(Yval)
    for i in range(Yval.shape[0]):
        Yp_val[i] = model.predict(Xval[[i], :])
    
    # Test prediction
    model.recalibrate(copy.deepcopy(Xtrain_[-calibration_window:]),
                      copy.deepcopy(Ytrain_[-calibration_window:]))
    ntest = Xtest.shape[0]
    Yp_test = np.zeros((ntest, 24))
    for i in range(ntest):
        Yp_test[i] = model.predict(Xtest[[i], :])
    
    val_pred_file = f"LEAR_{calibration_window}_val_predictions.csv"
    test_pred_file = f"LEAR_{calibration_window}_test_predictions.csv"
    pandas.DataFrame(Yp_val).to_csv(os.path.join(path, val_pred_file), index=False)
    pandas.DataFrame(Yp_test).to_csv(os.path.join(path, test_pred_file), index=False)

def recalibrate(calibration_window, country):
    path = get_path(country)
    recalibrated_predictions = evaluate_lear_in_test_dataset(
        path_recalibration_folder=path, path_datasets_folder=path,
        dataset=country, years_test=2, calibration_window=calibration_window,
        begin_test_date=None, end_test_date=None)

    # Remove the first 7 predictions because they are not part of the test set
    recalibrated_predictions = recalibrated_predictions.tail(-7)
    
    # Although another copy of the predictions are saved in evluate_lear_in_test_dataset, we
    # Save our own copy.
    test_recal_pred_file = f"LEAR_{calibration_window}_test_recalibrated_predictions.csv"
    recalibrated_predictions.to_csv(os.path.join(path, test_recal_pred_file), index=False)
