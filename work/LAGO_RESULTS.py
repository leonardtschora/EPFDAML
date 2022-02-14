import os
import work.parallel_scikit as ps
from joblib import Parallel, delayed
import itertools, numpy as np, time, pandas

import work.LAGO_RESULTS_UTILS_LEAR
import work.LAGO_RESULTS_UTILS_DNN
LRUL = work.LAGO_RESULTS_UTILS_LEAR
LRUD = work.LAGO_RESULTS_UTILS_DNN

os.environ["EPFDAML"] = os.curdir
########################################## Experimental set up
seeds = (1, 2, 3, 4)
countries = ["FR", "DE", "BE"]
nlayers = (1, 2, 3 ,4)
max_evals = 1000
shuffle_train = False
n_cpus = -1
calibration_windows = (56, 84, 1092, 1456)

########################################### DNN
all_combis = [k for k in itertools.product(countries, nlayers)]
already_done = {"seed_1" : [], "seed_2" : [], "seed_3" : [], "seed_4" : []}

# LAUNCH ALL THE ITERATIONS : COUNTRY/NLAYERS in parallel, SEEDs sequential
for seed in seeds:
    already = already_done["seed_" + str(seed)]
    combis = [k for k in itertools.filterfalse(lambda x: x in already , all_combis)]
    Parallel(n_jobs=n_cpus)(
        delayed(LRUD.optim_once)(iter_, max_evals, shuffle_train, seed)
        for iter_ in combis)
    
# Load the results : pick the best configuration for each run
all_combis_seeds = [k for k in itertools.product(seeds, countries, nlayers)]
df = LRUD.load_results(all_combis_seeds)

# Produces the validation and test predictions of the best configuration
seed_country = [k for k in itertools.product(seeds, countries)]
for (seed, country) in seed_country:
    LRUD.test_best_config(df, country, seed, shuffle_train)

########################################### LEAR    
# Produces the validation and test predictions of the LEAR models
nval = 362
calibration_windows = (56, 84, 1092, 1456)
lear_times = pandas.DataFrame(itertools.product(countries, calibration_windows),
                              columns=["country", "calibration_window"])
i = 0
for country, calibration_window in itertools.product(countries, calibration_windows):
    start = time.time()
    LRUL.predict_val_test(calibration_window, country, nval)
    stop = time.time()
    elapsed_time = (stop - start)
    lear_times.loc[i, "times"] = elapsed_time
    i += 1
    
lear_times.to_csv(os.path.join(
    os.environ["EPFDAML"], "data", "datasets", "lear_times.csv"), index=False)

########################################### RECALIBRATION
# Recalibrate in parallel = several recalibration at once, not on the same dataset
models = ["LEAR", "DNN"]
params = {"LEAR" : calibration_windows, "DNN" : seeds}
recalibrations = np.concatenate(
    [[k for k in itertools.product([model], countries, params[model])] for model in models])

def recalibrate(model, country, param, df):
    start = time.time()
    param = int(param)
    if model == "LEAR":
        LRUL.recalibrate(param, country)
    if model == "DNN":
        # THe shuffle train argument is just here to retrieve the result file!
        LRUD.recalibrate(param, country, df, False)
    stop = time.time()
    return stop - start
    
times = Parallel(n_jobs=n_cpus)(
    delayed(recalibrate)(model, country, param, df)
    for model, country, param in recalibrations)
res = pandas.DataFrame(recalibrations, columns=["models", "country", "params"])
res.loc[:, "times"] = times
res.to_csv(os.path.join(
    os.environ["EPFDAML"], "data", "datasets", "recalibration_times.csv"), index=False)
############################################# Re-Arrange the dates of the recalibrated predictions
def forecast_file(model, country, param, shuffle_train=False):
    if model == "DNN":
        losses = df.loc[np.logical_and(df.seed == int(param), df.country == country), :]
        indmin = losses.loss.values.argmin()
        best_nlayer = losses.iloc[indmin].nlayer
        
        return f'DNN_forecast_nl{best_nlayer}_dat{country}_YT2_SFH{shuffle_train}_CW4_LAGO_RESULTS_{param}.csv'
    if model == "LEAR":
        return f'LEAR_forecast_dat{country}_YT2_CW{param}.csv'


for model, country, param in recalibrations:
    path = get_path(country)
    data = pandas.read_csv(os.path.join(path, forecast_file(model, country, param)))
    data = data.tail(-7)
    try: data = data.drop(columns="Date")
    except: data = data.drop(columns="Unnamed: 0")
    test_recal_pred_file = f"{model}_{param}_test_recalibrated_predictions.csv"
    data.to_csv(os.path.join(path, test_recal_pred_file), index=False)
