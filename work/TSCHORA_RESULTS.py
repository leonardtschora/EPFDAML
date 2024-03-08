import pandas, copy, os, sys, tensorflow as tf
# Skip this line if your gpu is set up
os.environ["CUDA_VISIBLE_DEVICES"]="-1"

# Change this if you need to
os.environ["EPFDAML"] = os.curdir
sys.path.append(os.environ["EPFDAML"])

# import models
from work.models.Feature import Naive
from work.models.MLPWrapper import MLPWrapper
from work.models.CNNWrapper import LeNetWrapper
from work.models.SVR import MultiSVR, ChainSVR, SVR
from work.models.RFR import RFR
from work.TSCHORA_results_utils import run

"""
Run parameters. 
First, select 1 or more TASK to run : 
    GRID_SEARCH : performs hyper-parameter optimization using a random grid search method. This will produce a results.csv file for each model and country.
    LOAD_AND_TEST : provided that a results.csv file exists, this will load the best configuration, recompute the validation error and compute the test error.
    RECALIBRATE : provided that a results.csv file exists, this will load the best configuration and compute the error on the test set by recalibrating for each day.

Then, some general parameters have to be defined :
    name : a string that will be appended to the model's files (forecast, results, etc)
    dataset : a string indicating the dataset to load. Must of ("", "2" or "Frites"). Dataset "" will use the exact same datasets as in the epftoolbox. Dataset "2" is the enriched dataset on the same period. Dataset "Frites" is the datasets on the recent period.
    "EPF{dataset}_" + str(country)
    base_dataset_name : This is used for loading results from previous dataset. 
    For instance, we used base_dataset_name = "2" and dataset = "Frites"
    for using the best found configuration on the first period for recalibrating the second period.
    Enter the same name for disabling configuration loading.
    "n_val" : Interger indicating the number of validation samples to use.
    "models" : A list of model wrappers to evaluate
    "countries" : the region to evaluate

Then, according to the chosen tasks, some parameters have to be settled.
Here are the GRID SEARCH PARAMS
    "fast" : Boolean, If True, this will accelerate the training phase by stopping training of the models super-early (2 iterations for SVR, 2 epochs for model wrappers, 2 trees for random forests). This is usefull for testing the code integrity.
    "n_combis" : The number of combination to evaluate.
    "n_rep" : The number of repetitions. The total number of configurations is n_combis * n_rep.
    "restart" : Boolean indicating if the previously tested configuration must be erased or not.
    "stop_after" : Training time in seconds after when a model's training must be stopped. -1 will let the trainings finish naturally.
    "n_cpus" : The number of core to use. -1 will use all the cores.

The RECALIBRATION Parameters are the following :
    "n_shap" : The number of shap values coalitions to sample. 0 Means no shap values.
    "start" : interger. The first day of the test set to evaluate.
    "stop" : interger. The last day of the test set to evaluate.
    "calibration window" : interger. The length of the train set used during recalibration.
"""


kwargs = {
    # TASKS
    "GRID_SEARCH" : True,
    "LOAD_AND_TEST" : True,
    "RECALIBRATE" : False,   

    # GENERAL PARAMS
    "name" : "TSCHORA",
    "dataset" : "2",
    "base_dataset_name" : "2",
    "n_val" : 362,
    "models" : [MLPWrapper], # [LeNetWrapper, MLPWrapper, MultiSVR, ChainSVR, RFR],    
    "countries" : ["FRDE"], # "DE", "BE"],
    
    # GRID SEARCH PARAMS
    "fast" : True,
    "n_combis" : 2,
    "restart" : True,
    "n_rep" : 2,
    "stop_after" : -1,
    "n_cpus" : -1,

    # RECALIBRATION PARAMS
    "n_shap" : 0,
    "start" : 0,
    "stop" : 721,
    "calibration_window" : 1454,
}
run(**kwargs)

"""
# The used parameters were:

# PERIOD T1 : BASE DATASETS
kwargs = {
    # TASKS
    "GRID_SEARCH" : True,
    "LOAD_AND_TEST" : True,
    "RECALIBRATE" : True,   

    # GENERAL PARAMS
    "name" : "TSCHORA",
    "dataset" : "",
    "base_dataset_name" : "",
    "n_val" : 362,
    "models" : [LeNetWrapper, MLPWrapper, MultiSVR, ChainSVR, RFR],    
    "countries" : ["FR", "DE", "BE"],
    
    # GRID SEARCH PARAMS
    "fast" : False,
    "n_combis" : 200,
    "restart" : True,
    "n_rep" : 20,
    "stop_after" : 100,
    "n_cpus" : -1,

    # RECALIBRATION PARAMS
    "n_shap" : 0,
    "start" : 0,
    "stop" : 721,
    "calibration_window" : 4 * 362,
}
run(**kwargs)


# PERIOD T1 : ENRICHED DATASETS
kwargs = {
    # TASKS
    "GRID_SEARCH" : True,
    "LOAD_AND_TEST" : True,
    "RECALIBRATE" : False,   

    # GENERAL PARAMS
    "name" : "TSCHORA",
    "dataset" : "2",
    "base_dataset_name" : "2",
    "n_val" : 362,
    "models" : [LeNetWrapper, MLPWrapper, MultiSVR, ChainSVR, RFR],    
    "countries" : ["FR"],# , "DE", "BE", "FRDEBE"],
    
    # GRID SEARCH PARAMS
    "fast" : True,
    "n_combis" : 200,
    "restart" : True,
    "n_rep" : 20,
    "stop_after" : 100,
    "n_cpus" : -1,

    # RECALIBRATION PARAMS
    "n_shap" : 0,
    "start" : 0,
    "stop" : 721,
    "calibration_window" : 4 * 362,
}
run(**kwargs)


# PERIOD T2 : we use the configurations of T1 and thus base_dataset_name=2
kwargs = {
    # TASKS
    "GRID_SEARCH" : True,
    "LOAD_AND_TEST" : True,
    "RECALIBRATE" : True,   

    # GENERAL PARAMS
    "name" : "TSCHORA",
    "dataset" : "3",
    "base_dataset_name" : "2",
    "n_val" : 362,
    "models" : [LeNetWrapper, MLPWrapper, MultiSVR, ChainSVR, RFR],    
    "countries" : ["FR", "DE", "BE"],
    
    # GRID SEARCH PARAMS
    "fast" : False,
    "n_combis" : 200,
    "restart" : True,
    "n_rep" : 20,
    "stop_after" : 100,
    "n_cpus" : -1,

    # RECALIBRATION PARAMS
    "n_shap" : 0,
    "start" : 0,
    "stop" : 725,
    "calibration_window" : 1456,
}
# run(**kwargs)
"""