from work.analysis.evaluate import mae, smape, mape, rmse, rmae, dae, cmap
from work.models.model_wrapper import ModelWrapper
from work.models.MLPWrapper import MLPWrapper
from work.models.SVR import MultiSVR, ChainSVR
from work.models.RFR import RFR
from work.models.CNNWrapper import LeNetWrapper
from work.models.LAGO_wrapper import LAGOWrapper
import numpy as np, pandas, copy, itertools, matplotlib
import matplotlib.pyplot as plt

import work.analysis.metrics_utils
import work.analysis.evaluate
mu = work.analysis.metrics_utils

datasets = ("validation", "test", "test_recalibrated")
versions = ("", "2", "3")
countries = ("FR", "DE", "BE")
models = (MLPWrapper, RFR, ChainSVR, MultiSVR, LAGOWrapper)
metrics = (smape, mae, dae, rmae)
lago_params = {"LEAR" : (56, 84, 1092, 1456), "DNN" : (1, 2, 3, 4)}

predictions, model_wrappers = mu.load_forecasts(
    datasets, countries, models, versions, lago_params)
real_prices, naive_forecasts = mu.load_real_prices(countries)
results = mu.compute_metrics(predictions, model_wrappers, metrics, real_prices,
                             naive_forecasts)
dms = mu.compute_pvalues(predictions, model_wrappers, real_prices)

# All version individually
version = ""
res_val = mu.plot_scaled_metrics(results, model_wrappers, "validation",
                                 version, metrics)
res_test = mu.plot_scaled_metrics(results, model_wrappers, "test", version, metrics)
res_test_recalibrated = mu.plot_scaled_metrics(
    results, model_wrappers, "test_recalibrated", version, metrics)

version = "2"
res_val_2 = mu.plot_scaled_metrics(results, model_wrappers, "validation",
                                      version, metrics)
res_test_2 = mu.plot_scaled_metrics(results, model_wrappers, "test", version,
                                       metrics)
res_test_recalibrated_2 = mu.plot_scaled_metrics(
    results, model_wrappers, "test_recalibrated", version, metrics)


version = "3"
res_val_3 = mu.plot_scaled_metrics(results, model_wrappers, "validation",
                                      version, metrics)
res_test_3 = mu.plot_scaled_metrics(results, model_wrappers, "test", version,
                                       metrics)
res_test_recalibrated_3 = mu.plot_scaled_metrics(
    results, model_wrappers, "test_recalibrated", version, metrics)

# Compare versions
res = mu.plot_summary_multi(res_test_recalibrated_2, res_test_recalibrated_3)

val_v1_v2 = mu.plot_diff(res_val, res_val_2, ["validation_1", "validation_2"])
val_v2_v3 = mu.plot_diff(res_val_2, res_val_3, ["validation_2", "validation_3"])
val_v1_v3 = mu.plot_diff(res_val, res_val_3, ["validation_1", "validation_3"])
mu.plot_summary(res_val, res_val_2, res_val_3)

test_v1_v2 = mu.plot_diff(res_test, res_test_2, ["test_1", "test_2"])
test_v2_v3 = mu.plot_diff(res_test_2, res_test_3, ["test_2", "test_3"])
test_v1_v3 = mu.plot_diff(res_test, res_test_3, ["test_1", "test_3"])
mu.plot_summary(res_test, res_test_2, res_test_3)

recalibrated_test_v1_v2 = mu.plot_diff(
    res_test_recalibrated, res_test_recalibrated_2,
    ["test_recalibrated_1", "test_recalibrated_2"])
recalibrated_test_v2_v3 = mu.plot_diff(
    res_test_recalibrated_2, res_test_recalibrated_3,
    ["test_recalibrated_2", "test_recalibrated_3"])
recalibrated_test_v1_v3 = mu.plot_diff(
    res_test_recalibrated, res_test_recalibrated_3,
    ["test_recalibrated_1", "test_recalibrated_3"])
mu.plot_summary(res_test_recalibrated, res_test_recalibrated_2,
                res_test_recalibrated_3)

best_metrics =  mu.plot_best_metrics(res_test, res_test_recalibrated, countries,
                                     metrics, model_wrappers, nmw, columns, index)

## PValues


version = "2"
dataset = "test_recalibrated"
versions = {"" : "BASE", "2" : "ENRICHED"}
dms, ms = mu.compute_pvalues_cross(predictions, model_wrappers,
                                   real_prices, versions)
mu.plot_DM_test_cross(dms, ms)
mu.print_DM_cross(dms, ms)

dms = mu.compute_pvalues(predictions, model_wrappers, real_prices)
mu.plot_DM_test(dms, model_wrappers, dataset, version)
