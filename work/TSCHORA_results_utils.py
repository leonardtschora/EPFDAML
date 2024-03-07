from work.models.MLPWrapper import MLPWrapper
from work.models.CNNWrapper import LeNetWrapper
from work.models.LAGO_wrapper import LAGOWrapper
from work.models.SVR import MultiSVR, ChainSVR, SVR
from work.models.RFR import RFR
from work.columns import labels_and_pos
import copy, numpy as np, time, pandas, copy
from work.models.Splitter import MySplitter
import work.parallel_scikit as ps

def run(**kwargs):
    # TASKS
    GRID_SEARCH = kwargs["GRID_SEARCH"]
    LOAD_AND_TEST = kwargs["LOAD_AND_TEST"]
    RECALIBRATE = kwargs["RECALIBRATE"]

    # GENERAL PARAMS
    name = kwargs["name"]
    dataset = kwargs["dataset"]
    base_dataset_name = kwargs["base_dataset_name"]    
    n_val = kwargs["n_val"]    
    models = kwargs["models"]
    countries = kwargs["countries"]

    # GRID SEARCH PARAMS    
    fast = kwargs["fast"]
    n_combis = kwargs["n_combis"]
    restart = kwargs["restart"]
    n_rep = kwargs["n_rep"]    
    stop_after = kwargs["stop_after"]
    n_cpus = kwargs["n_cpus"]

    # RECALIBRATION PARAMS
    n_shap = kwargs["n_shap"]
    start_rec = kwargs["start"]
    stop_rec = kwargs["stop"]
    calibration_window= kwargs["calibration_window"]    

    # Some init
    recalibration_times = pandas.DataFrame(columns=["country", "model", "times"])   
    restart_changed = False
    for country in countries:
        for model in models:
            start = time.time()
            if restart_changed:
                restart_changed = False
                restart = True
                
            if GRID_SEARCH:
                for i in range(n_rep):
                    model_wrapper = run_grid_search(
                        name, dataset, model, country, base_dataset_name, fast,
                        n_combis, restart, stop_after, n_cpus, n_val)

                    if (n_rep > 1) and restart:
                        print("Step specified but wram start is not allowed.")
                        print("Disabling restart.")
                        restart_changed = True
                        restart = False
                    
                    df = model_wrapper.load_results()
                    best_params = model_wrapper.best_params(df)
                    print(f"LOSS = {best_params['maes']}")
                    print(f"TIME = {round((time.time() - start)  / 3600, ndigits=2)}h")
                    
            if LOAD_AND_TEST:
                pause = time.time()
                run_load_and_test(name, dataset, model, country, n_val,
                                  base_dataset_name)
                load_and_test_time = time.time() - pause
                print(
                    f"LOAD AND TEST TIME = {round(load_and_test_time, ndigits=2)}s")
                print(
                    f"ESTIMATED RECALIBRATION TIME = {round(load_and_test_time * (stop_rec - start_rec) / 3600, ndigits=2)}h ON 1 CPU")                
                start = start - (time.time() - pause)
            
            if RECALIBRATE:
                pause = time.time()
                total_time = run_recalibrate(name, dataset, model, country, n_cpus,
                                             start_rec, stop_rec, n_shap, n_val,
                                             base_dataset_name, calibration_window)
                recalibration_times = pandas.concat([
                    pandas.DataFrame(
                    {"country" : country,
                    "model" : get_model_string(model),
                    "times" : total_time}, index=[0]),
                    recalibration_times], ignore_index=True)

                # recalibration_times = recalibration_times.append(
                #     {"country" : country,
                #      "times" : total_time,
                #      "model" : get_model_string(model)},
                #     ignore_index=True)

                start = start - (time.time() - pause)

    recalibration_times.to_csv("recalibrattion_times" + str(time.time()) + ".csv")
    
def run_grid_search(name, dataset, model, country, base_dataset_name, fast,
                    n_combis, restart, stop_after, n_cpus, n_val):
    spliter = MySplitter(n_val, shuffle = False)    
    model_wrapper, validation_mode, external_spliter = create_mw(
        country, dataset, model, name, spliter=spliter)
    
    X, y = model_wrapper.load_train_dataset()
    n = X.shape[0]
    if base_dataset_name == dataset: load_from = None
    else: load_from = base_dataset_name
    search_space = get_search_space(
        model_wrapper, n, country, dataset=dataset, fast=fast,
        load_from=load_from, stop_after=stop_after)
    
    # This makes sure that all the models will have the same sampled parameters
    ps.set_all_seeds(1)
    param_list, seeds = ps.get_param_list_and_seeds(
        search_space, n_combis, country,
        model_wrapper=model_wrapper, restart=restart)
    # param list is a list of dictionaries with the parameters to be tested
    # seeds is a list of seeds to be used for each parameter combination
    # n_combis is the number of parameter combinations to be tested
    results = ps.parallelize(n_cpus, model_wrapper, param_list, X, y,
                             seeds=seeds, validation_mode = validation_mode,
                             external_spliter = external_spliter)
    df = ps.results_to_df(results, param_list, seeds=seeds, n_cpus=n_cpus,
                          map_dict=model_wrapper.map_dict(), cv=1)
    if not restart:
        # Dont use load results here because it will parse string as python objects!
        # df = pandas.read_csv(model_wrapper.results_path()).append(df)
        df = pandas.concat([pandas.read_csv(model_wrapper.results_path()), df])
    df.to_csv(model_wrapper.results_path(), index=False)
    return model_wrapper

def run_load_and_test(name, dataset, model, country, n_val, base_dataset_name):    
    spliter = MySplitter(n_val, shuffle = False)
    model_wrapper, validation_mode, external_spliter = create_mw(
        country, dataset, model, name, spliter=spliter)

    base_model_wrapper = copy.deepcopy(model_wrapper)
    base_model_wrapper.dataset_name = base_model_wrapper.dataset_name.replace(
        dataset, base_dataset_name)

    # Load the best params from this mw
    df = base_model_wrapper.load_results()
    best_params = base_model_wrapper.best_params(df)
    n_combis = df.shape[0]
    print(f"EVALUATED {n_combis} COMIBINATIONS ON DATASET '{base_dataset_name}'")
    print(f"ORIGINAL VAL SET MAE = {round(best_params['maes'], ndigits=2)}")
    print(f"ORIGINAL TRAINING TIME = {round(best_params['times'], ndigits=2)}s")    

    best_params = base_model_wrapper.best_params(df, for_recalibration=True)
    
    # Generate Validation and test metrics
    regr = model_wrapper.test_and_save_epf(
        best_params, seed=best_params["seeds"], validation_mode=validation_mode,
        external_spliter=external_spliter)

def run_recalibrate(name, dataset, model, country, n_cpus, start, stop, n_shap,
                    n_val, base_dataset_name, calibration_window):
    spliter = MySplitter(n_val, shuffle = True)    
    model_wrapper, validation_mode, external_spliter = create_mw(
        country, dataset, model, name, spliter=spliter)

    base_model_wrapper = copy.deepcopy(model_wrapper)
    base_model_wrapper.dataset_name = base_model_wrapper.dataset_name.replace(
        dataset, base_dataset_name)
    
    # Load the best params from this mw
    print(f"LOADING THE BEST CONFIGURATION FROM DATASET '{base_dataset_name}'")
    df = base_model_wrapper.load_results()
    best_params = base_model_wrapper.best_params(df, for_recalibration=True)
    
    # Recalibrate
    total_time = model_wrapper.recalibrate_epf(
        seed=best_params["seeds"], ncpus=n_cpus,
        calibration_window=calibration_window,
        best_params=best_params, n_shap=n_shap, start=start, stop=stop)
    return total_time
    
def get_date_cols(cc, version):
    if version == "":
        return [str(get_country_code(cc)) + "_day_of_week"]
    # TODO : ajouter pour FRDEBE
    if cc == "FRDEBE": cc = "FR"
    return [f'{cc}_day_1', f'{cc}_day_2', f'{cc}_day_of_week_1',            
            f'{cc}_day_of_week_2', f'{cc}_week_1', f'{cc}_week_2',
            f'{cc}_month_1', f'{cc}_month_2']

def create_mw(country, dataset, model, name, spliter=None):
    dataset_ = f"EPF{dataset}_" + str(country)
    labels = get_labels(country, dataset)
    date_col = get_date_cols(country, dataset)
    data_cols = get_col_names(country, dataset)
    cc = get_country_code(country)           
    name_ = get_model_string(model) + "_" + name

    return create_model_wrapper(
        model, name_, dataset_, labels, date_col, data_cols,
        spliter, country)    
    
def create_model_wrapper(model, name, dataset, labels, date_col, data_cols,
                         spliter, country):
    if (model is LeNetWrapper):
        model_wrapper = model(
            name, dataset, labels, date_col, data_cols, spliter=spliter)
        validation_mode = "internal"
        external_spliter = None
    else:
        if model is MLPWrapper:
            model_wrapper = model(
                name, dataset, labels, spliter=spliter)
            validation_mode = "internal"
            external_spliter = None                
        else:
            model_wrapper = model(name, dataset, labels)
            validation_mode = "external"
            external_spliter = spliter
    return model_wrapper, validation_mode, external_spliter

def get_search_space(model_wrapper, n, country, stop_after=-1,
                     dataset="", fast=False, load_from=None):
    if load_from is None:
        return model_wrapper.get_search_space(
            country=country, n=n, fast=fast, stop_after=stop_after)

    # Load results from the original version
    base_model_wrapper = copy.deepcopy(model_wrapper)
    base_model_wrapper.dataset_name = base_model_wrapper.dataset_name.replace(dataset, load_from)
    return base_model_wrapper.load_results()
    
def get_labels(country, dataset):
    # TODO : modifier pour FRDE, DEBE, FRBE
    if dataset not in ("FRBL8", "FRBL10", "FRBL11"):
        if country == "FRDEBE":
            labels = [f"FR_price_{i}" for i in range(24)] + [f"DE_price_{i}" for i in range(24)] + [f"BE_price_{i}" for i in range(24)]
        # elif country == "FRDE":
            # TODO : ajouter les labels pour FRDE
        else: labels = [f"{get_country_code(country)}_price_{i}" for i in range(24)]
    else:
        labels = [f"{get_country_code(country)}_price", ]
    return labels

def get_col_names(country, version):
    model_wrapper = RFR("", f"EPF{version}_{country}", get_labels(country, version))
    return labels_and_pos(model_wrapper, version, country, for_plot=False)[0]

def get_col_names_(country):
    if country == "FR":
        cols = ("Generation forecast", "System load forecast")
    if country == "DE":
        cols = ("Ampirion Load Forecast", "PV+Wind Forecast")
    if country == "BE":
        cols = ("Generation forecast", "System load forecast")
    if country == "NP":
        cols = ("Grid load forecast", "Wind power forecast")
    if country == "PJM":
        cols = ("System load forecast", "Zonal COMED load foecast")
    if country == "FRDEBE":
        cols = get_col_names_("FR") + get_col_names_("DE") + get_col_names_("BE")
    # TODO : ajouter pour FRDE, DEBE, etc.
    return cols

def get_country_code(country):
    # TODO : ajouter les codes pour FRDE, DEBE, FRBE
    if country == "FR":
        code = "FR"
    if country == "FRDEBE":
        code = "FRDEBE"        
    if country == "DE":
        code = "DE"
    if country == "BE":
        code = "BE"
    if country == "NP":
        code = "NOR"
    if country == "PJM":
        code = "US"
    return code

def get_model_string(model):
    s = ""
    if model is RFR:
        s = "RFR"
    if model is SVR:
        s = "SVR"
    if model is ChainSVR:
        s = "SVRChain"
    if model is MultiSVR:
        s = "SVRMulti"        
    if model is MLPWrapper:
        s = "MLP"
    if model is LeNetWrapper:
        s = "CNN"
    if model is LAGOWrapper:
        s = ""
    return s
