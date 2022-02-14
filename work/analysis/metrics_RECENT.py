from work.analysis.evaluate import mae, smape, mape, rmse, rmae, dae, cmap
from work.models.model_wrapper import ModelWrapper
from work.models.MLPWrapper import MLPWrapper
from work.models.SVR import MultiSVR, ChainSVR
from work.models.RFR import RFR
from work.models.Feature import Naive
from work.models.CNNWrapper import LeNetWrapper
from work.models.LAGO_wrapper import LAGOWrapper
import numpy as np, pandas, copy, itertools, matplotlib
import matplotlib.pyplot as plt
from work.models.Splitter import MySplitter

import work.analysis.metrics_utils
import work.analysis.evaluate
mu = work.analysis.metrics_utils

datasets = ("validation", "test", "test_recalibrated")
versions = ("3", )
countries = ("FR", "DE", "BE")
models = (LeNetWrapper, MLPWrapper, ChainSVR, MultiSVR, RFR)
metrics = (smape, mae, dae, rmae)

predictions, model_wrappers = mu.load_forecasts(
    datasets, countries, models, versions, {})
real_prices, naive_forecasts = mu.load_real_prices(countries, versions[0])
results = mu.compute_metrics(predictions, model_wrappers, metrics, real_prices,
                             naive_forecasts)
dms = mu.compute_pvalues(predictions, model_wrappers, real_prices)

# All version individually
version = "3"
res_val = mu.plot_scaled_metrics(results, model_wrappers, "validation",
                                 version, metrics)
res_test = mu.plot_scaled_metrics(results, model_wrappers, "test",
                                  version, metrics)
res_test_recalibrated = mu.plot_scaled_metrics(
    results, model_wrappers, "test_recalibrated", version, metrics)

res_val.columns = [c + "_val" for c in res_val.columns]
res_test.columns = [c + "_test" for c in res_test.columns]
final = mu.plot_summary(res_val, res_test, res_test_recalibrated)
