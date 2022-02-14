import numpy as np, pandas, copy, itertools, math, copy
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from work.analysis.evaluate import mae, smape, mape, rmse, rmae, dae, cmap, cmap_2
from work.TSCHORA_results_utils import create_model_wrapper, get_model_string, get_country_code
from work.analysis.evaluate import cmap_scaled_values, cmap_diff_values, cmap_diff_values_2
from work.analysis.evaluate import load_prevs_mw
from work.models.LAGO_wrapper import LAGOWrapper
from work.models.CNNWrapper import LeNetWrapper
from work.models.Feature import Naive
from work.models.Splitter import MySplitter
from epftoolbox.evaluation import DM
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

class MinAbsScaler(TransformerMixin, BaseEstimator):
    """
    Standardize the data using a Standard Scaler, then apply the arcsinh.
    """
    def __init__(self, epsilon=10e-5):
        self.epsilon = epsilon

    def fit(self, X, y=None):
        X = check_array(X, accept_sparse=True)
        self.n_features_ = X.shape[1]

        min_abs = np.abs(np.array(X)).min()
        min_abs = np.clip(min_abs, a_min=self.epsilon, a_max=None)
        self.min_abs = min_abs.reshape(-1, 1)
        
        self.is_fitted_ = True
        return self        
    
    def transform(self, X):
        check_is_fitted(self, 'n_features_')
        X = check_array(X, accept_sparse=True)        
        if X.shape[1] != self.n_features_:
            raise ValueError('Shape of input is different from what was seen'
                             'in `fit`')
        
        transformed_data = np.array(X) / self.min_abs
        return transformed_data

    def inverse_transform(self, X, y=None):
        check_is_fitted(self, 'n_features_')
        X = check_array(X, accept_sparse=True)        
        if X.shape[1] != self.n_features_:
            raise ValueError('Shape of input is different from what was seen'
                             'in `fit`')
        
        transformed_data = np.array(X) * self.min_abs
        return transformed_data

def print_DM_cross(dms, ms, versions):
    dfs = {}
    vkeys = [k for k in versions.keys()]
    for (k, country) in enumerate(ms.keys()):
        df = pandas.DataFrame()
        enriched_ind = np.where(ms[country][:, 0] == vkeys[0])[0]
        multi_ind = np.where(ms[country][:, 0] == vkeys[1])[0]

        df["Model"] = [k.string() for k in ms[country][enriched_ind, 1]]
        df[f"Pvalue({versions[vkeys[0]]}, {versions[vkeys[1]]})"] = np.round(dms[k, enriched_ind, multi_ind], decimals=4)
        df[f"Pvalue({versions[vkeys[1]]}, {versions[vkeys[0]]})"] = np.round(dms[k, multi_ind, enriched_ind], decimals=4)
        dfs[country] = df
    return dfs
        
def plot_DM_test_cross(dms, ms, version):
    countries = [k for k in ms.keys()]
    fig = plt.figure(figsize=(19.2, 10.8))
    gs1 = gridspec.GridSpec(1, 3, figure=fig, wspace=0.4)
    ax1 = fig.add_subplot(gs1[:, 0])
    ax2 = fig.add_subplot(gs1[:, 1], sharex=ax1)
    ax3 = fig.add_subplot(gs1[:, 2], sharex=ax1)

    axes = np.array([ax1, ax2, ax3])
    for i, country in enumerate(countries):
        data = dms[i, :, :]
        columns = [k.string() for k in ms[country][:, 1]]
        nmw = len(columns)
        ax = axes[i]
        data = np.where(np.isnan(data), 1, data)
        im = ax.imshow(data, cmap=cmap(), vmin=0, vmax=0.1)

        labels_fontsize = 14
        cols_to_display = [c.split(" ")[-1] for c in columns]
        ax.set_xticks(range(len(cols_to_display)))
        ax.set_xticklabels([])
        for x, col in enumerate(cols_to_display):
            ax.text(x + 0.25, nmw - 0.5, col, rotation=45, fontsize=labels_fontsize, ha="right", va="top")
        
        ax.set_yticks(range(len(cols_to_display)))
        ax.set_yticklabels(cols_to_display, fontsize=labels_fontsize)
        ax.plot(range(len(cols_to_display)), range(len(cols_to_display)), 'wx')
        ax.set_title(country)

        i_masks = (2, 4, 6)
        x0 = -0.23
        for j in range(nmw):                       
            if j in i_masks:
                yaxes = 1 - (j / nmw)            
                ax.annotate("",
                            xy=[x0, yaxes],
                            xycoords="axes fraction",
                            xytext = [0, yaxes],                         
                            textcoords="axes fraction",
                            arrowprops={"arrowstyle": "-", "linestyle" : "--",
                                        "linewidth" : 1, "color" : "k"})
                xaxes = 0.01 + j / nmw
                xratio = 1.25
                y0 = -0.15
                ax.annotate("",
                            xy=[xaxes, 0],
                            xycoords="axes fraction",
                            xytext = [xaxes - xratio / nmw, y0],
                            textcoords="axes fraction",
                            arrowprops={"arrowstyle": "-", "linestyle" : "--",
                                        "linewidth" : 1, "color" : "k"})
                ax.annotate("",
                            xy=[xaxes - xratio / nmw, y0],
                            xycoords="axes fraction",
                            xytext = [xaxes - xratio / nmw, y0-0.05],
                            textcoords="axes fraction",
                            arrowprops={"arrowstyle": "-", "linestyle" : "--",
                                        "linewidth" : 1, "color" : "k"})
        yindices = (3, 7)
        labels = ("SVR", "SVR")
        for yind, label in zip(yindices, labels):
            ax.text(-2.5, yind, label, ha="center", size=labels_fontsize, rotation=90)

        xindices = (1.5, 5.5)
        labels = ("SVR", "SVR")
        for xind, label in zip(xindices, labels):
            pad = 1.5
            ax.text(xind, nmw + pad, label, ha="center", size=labels_fontsize)

        xindices = (0.5, 4.5)
        labels = version.values()
        pad = 2.25
        for xind, label in zip(xindices, labels):
            ax.text(xind, nmw + pad, label, ha="center", size=labels_fontsize, c="r")

        yindices = (2.5, 6.5)
        labels = version.values()
        pad = 3.25
        for yind, label in zip(yindices, labels):
            ax.text(0 - pad, yind, label, ha="center", size=labels_fontsize, c="r", rotation=90)            
            
        # Separate versions
        xsep = 0.5
        ysep = 0.5
        ax.annotate("",
                    xy=[-0.45, ysep],
                    xycoords="axes fraction",
                    xytext = [1, ysep],                         
                    textcoords="axes fraction",
                    arrowprops={"arrowstyle": "-", "linestyle" : "--",
                                "linewidth" : 3, "color" : "r"})
        ax.annotate("",
                    xy=[xsep, 0],
                    xycoords="axes fraction",
                    xytext = [xsep, 1],                         
                    textcoords="axes fraction",
                    arrowprops={"arrowstyle": "-", "linestyle" : "--",
                                "linewidth" : 3, "color" : "r"})
        xratio = 1.2
        ax.annotate("",
                    xy=[xsep, 0],
                    xycoords="axes fraction",
                    xytext = [xsep - xratio / nmw, y0],
                    textcoords="axes fraction",
                    arrowprops={"arrowstyle": "-", "linestyle" : "--",
                                "linewidth" : 3, "color" : "r"})
        ax.annotate("",
                    xy=[xsep - xratio / nmw, y0],
                    xycoords="axes fraction",
                    xytext = [xsep - xratio / nmw, y0-0.2],
                    textcoords="axes fraction",
                    arrowprops={"arrowstyle": "-", "linestyle" : "--",
                                "linewidth" : 3, "color" : "r"})        
            
        ax.tick_params(axis="y", pad=-0.1)
        ax.set_xticklabels([])
        ax.set_xticks([c + 0.5 for c in range(len(columns))], minor=True)
        ax.set_yticks([c + 0.45 for c in range(len(columns))], minor=True)
        ax.tick_params(which="minor", length=0)
        ax.grid("on", which="minor", linestyle="--", c="grey")

    
    cbar = plt.colorbar(im, ax=axes, orientation="horizontal", fraction=0.05, pad=0.21)
    cbar.ax.set_xlabel("Pvalue of the DM test")
        
    #plt.show()
    
    
def plot_DM_test(dms, model_wrappers, dataset, version):
    countries = [k for k in model_wrappers[version][dataset].keys()]
    columns, ind_lago = format_cols(model_wrappers[version][dataset][countries[0]])

    factor = 1
    fig = plt.figure(figsize=(factor * 19.2, factor * 10.8))
    gs0 = gridspec.GridSpec(2, 1, figure=fig, hspace=0.3)
    gs1 = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs0[0])
    gs2 = gridspec.GridSpecFromSubplotSpec(1, 6, subplot_spec=gs0[1])

    ax1 = fig.add_subplot(gs1[:, 0])
    ax2 = fig.add_subplot(gs1[:, 1], sharex=ax1)
    ax3 = fig.add_subplot(gs1[:, 2], sharex=ax1)

    ax4 = fig.add_subplot(gs2[:, 1:3])
    ax5 = fig.add_subplot(gs2[:, 3:5], sharex=ax4)
    axes = np.array([ax1, ax2, ax3, ax4, ax5])

    nmw = 12
    for i, country in enumerate(countries):
        data = dms[version][dataset][i, :, :]        
        ax = axes[i]
        data = np.where(np.isnan(data), 1, data)
        im = ax.imshow(data, cmap=cmap_2(), vmin=0, vmax=0.051)

        labels_fontsize = 14
        cols_to_display = [c.split(" ")[-1] for c in columns]
        ax.set_xticks(range(len(cols_to_display)))
        for x, col in enumerate(cols_to_display):
            ax.text(x + 0.25, nmw + 0.5, col, rotation=45, fontsize=labels_fontsize, ha="right", va="top")
        
        ax.set_yticks(range(len(cols_to_display)))
        ax.set_yticklabels(cols_to_display, fontsize=labels_fontsize)
        ax.plot(range(len(cols_to_display)), range(len(cols_to_display)), 'wx')
        ax.set_title(country)

        i_masks = (3, 5, 9)
        x0 = -0.23
        for j in range(nmw):                       
            if j in i_masks:
                yaxes = 1 - (0.92*j / nmw)            
                ax.annotate("",
                            xy=[x0, yaxes],
                            xycoords="axes fraction",
                            xytext = [0, yaxes],                         
                            textcoords="axes fraction",
                            arrowprops={"arrowstyle": "-", "linestyle" : "--",
                                        "linewidth" : 1, "color" : "k"})
                xaxes = 0.01 + 0.92*j / nmw
                xratio = 1.75
                y0 = -0.15
                ax.annotate("",
                            xy=[xaxes, 0],
                            xycoords="axes fraction",
                            xytext = [xaxes - xratio / nmw, y0],
                            textcoords="axes fraction",
                            arrowprops={"arrowstyle": "-", "linestyle" : "--",
                                        "linewidth" : 1, "color" : "k"})
                ax.annotate("",
                            xy=[xaxes - xratio / nmw, y0],
                            xycoords="axes fraction",
                            xytext = [xaxes - xratio / nmw, y0-0.05],
                            textcoords="axes fraction",
                            arrowprops={"arrowstyle": "-", "linestyle" : "--",
                                        "linewidth" : 1, "color" : "k"})

                
        yindices = (4.0, 7.1, 11.4)
        labels = ("SVR", "LEAR", "DNN")
        for yind, label in zip(yindices, labels):
            ax.text(-3, yind, label, ha="center", size=labels_fontsize, rotation=90)

        xindices = (1.8, 4.8, 8.5)
        labels = ("SVR", "LEAR", "DNN")
        for xind, label in zip(xindices, labels):
            pad = 2.9
            ax.text(xind, nmw + pad, label, ha="center", size=labels_fontsize)
            
        ax.tick_params(axis="y", pad=-0.1)

        ax.set_xticklabels([])
        ax.set_xticks([c + 0.5 for c in range(len(columns))], minor=True)
        ax.set_yticks([c + 0.45 for c in range(len(columns))], minor=True)
        ax.tick_params(which="minor", length=0)
        ax.grid("on", which="minor", linestyle="--", c="grey")
            
    if len(countries) != 3:
        cbar = plt.colorbar(im, ax=axes[-1], extend="max")
        cbar.ax.set_ylabel("Pvalue of the DM test")
        cbar.ax.tick_params(labelsize=labels_fontsize)
    else:
        cbar = plt.colorbar(im, ax=axes, orientation="horizontal", fraction=0.05)
        cbar.ax.set_xlabel("Pvalue of the DM test")
    fig.suptitle(f"DM test PVALUES for the recalibrated forecasts of the Test set")
    
    dfdata = {"Country" : [],
              "Model 1" : [],
              "Model 2" : [],
              "P Value 1" : [],
              "P Value 2" : []}
    for i, country in enumerate(countries):
        data = np.round(dms[version][dataset][i, :, :], decimals=3)
        print(data)
        model_comb = [k for k in itertools.combinations(
            range(data.shape[0]), 2)]
        m1s = [k[0] for k in model_comb]
        m2s = [k[1] for k in model_comb]
        mstr = [m.string() for m in  model_wrappers[version][dataset][country]]
        for (m1, m2) in zip(m1s, m2s):
            dfdata["Country"].append(country)
            dfdata["Model 1"].append(mstr[m1])
            dfdata["Model 2"].append(mstr[m2])
            dfdata["P Value 1"].append(data[m1, m2])
            dfdata["P Value 2"].append(data[m2, m1])
    df = pandas.DataFrame(dfdata)    
    return df

def compute_pvalues_cross(predictions, model_wrappers, real_prices, versions):
    dataset = "test_recalibrated"
    countries = ("FR", "DE", "BE")
    versions = versions.keys()
    models = np.concatenate([[(version, mw) for mw in model_wrappers[version][dataset][countries[0]] if ((type(mw) is not LAGOWrapper) and (type(mw) is not LeNetWrapper))] for version in versions])
    n = len(models)  
    dms = np.zeros((len(countries), n, n))
    ms = {}
    for (k, country) in enumerate(countries):
        models = np.concatenate([[(version, mw) for mw in model_wrappers[version][dataset][country] if ((type(mw) is not LAGOWrapper) and (type(mw) is not LeNetWrapper))] for version in versions])
        ms[country] = models
        n = len(models)    
        for (i, (v1, m1)) in enumerate(models):
            for (j, (v2, m2)) in enumerate(models):
                if i == j:
                    dms[k, i, j] = np.nan
                else:
                    p1 = predictions[v1][dataset][country][m1]
                    p2 = predictions[v2][dataset][country][m2]
                    if p1 is None or p2 is None:
                        dms[k, i, j] = np.nan
                    else:
                        dms[k, i, j] = DM(
                            p_real=real_prices[dataset][country],
                            p_pred_1=p1, p_pred_2=p2,
                            norm=1, version='multivariate')
    return dms, ms

def compute_pvalues(predictions, model_wrappers, real_prices):
    versions = [k for k in predictions.keys()] 
    dms = {}
    for version in versions:
        datasets = [k for k in predictions[version].keys()]         
        dms[version] = {}
        for dataset in datasets:
            countries = [k for k in predictions[version][dataset].keys()]
            if len(countries) > 0:
                nmw = len(model_wrappers[version][dataset][countries[0]])
                dms[version][dataset] = np.zeros((len(countries), nmw, nmw))
                for (k, country) in enumerate(countries):
                    for (i, m1) in enumerate(model_wrappers[version][dataset][country]):
                        for (j, m2) in enumerate(model_wrappers[version][dataset][country]):
                            if i == j: dms[version][dataset][k, i, j] = np.nan
                            else:
                                p1 = predictions[version][dataset][country][m1]
                                p2 = predictions[version][dataset][country][m2]
                                if p1 is None or p2 is None:
                                    dms[version][dataset][k, i, j] = np.nan
                                else:
                                    dms[version][dataset][k, i, j] = DM(
                                        p_real=real_prices[dataset][country],
                                        p_pred_1=p1, p_pred_2=p2,
                                        norm=1, version='multivariate')
    return dms

def plot_summary(res_1, res_2, res_3):
    final = res_1.join(res_2, how = "inner").join(res_3, how="inner")
    indices = final.index
    countries = np.unique(np.array([c[0] for c in indices]))
    metrics = np.unique(np.array([c[1] for c in indices]))
    
    ind_lago = np.array([("DNN " in i) or ("LEAR" in i) for i in final.columns])
    final = sort_columns(final, ind_lago)

    plot_matrix(
        final, data_info="Scaled metrics",
        title_info=f"All 3 versions",
        colormap=cmap_scaled_values(), scaler_class=MinAbsScaler)

    return final

def plot_summary_multi(res_1, res_2):
    final = res_1.join(res_2, how = "inner")
    indices = final.index
    countries = np.unique(np.array([c[0] for c in indices]))
    metrics = np.unique(np.array([c[1] for c in indices]))
    
    ind_lago = np.array([("DNN " in i) or ("LEAR" in i) for i in final.columns])
    final = sort_columns(final, ind_lago)

    plot_matrix(
        final, data_info="Scaled metrics",
        title_info=f"Models and their multi country counterpart",
        colormap=cmap_scaled_values(), scaler_class=MinAbsScaler)

    return final

def sort_columns(res, ind_lago):
    columns = res.columns
    indices = res.index
    
    columns_lago = copy.deepcopy(columns[ind_lago])
    ind_columns = np.argsort(columns[np.logical_not(ind_lago)])
    ind_columns = np.concatenate(
        (ind_columns,
         np.array([i for i in range(
             len(ind_columns), len(ind_columns) + len(columns_lago))], int)))
    sorted_columns = np.concatenate(
        (columns[np.logical_not(ind_lago)], columns_lago))[ind_columns]
        
    res = pandas.DataFrame(res.values, columns=columns, index=indices)
    res = res.loc[:, sorted_columns]
    return res

def plot_matrix(data, title_info="", data_info="", colormap=cmap_scaled_values(),
                scaler_class=MinAbsScaler, vmin=None, vmax=None):
    data = copy.deepcopy(data)
    
    # Normalize line by line
    if scaler_class is not None:
        for i in range(data.shape[0]):
            scaler = scaler_class()
            values = data.iloc[i].values.reshape(-1, 1)
            worse = values.max()
            values = np.where(values < 0, worse, values)
            data.iloc[i] = scaler.fit_transform(values).reshape(-1)
    plt.figure(figsize=(19.2, 10.8))
    dt = plt.imshow(data, cmap=colormap, vmin=vmin, vmax=vmax)
    cbar = plt.colorbar(dt, extend="both")
    #cbar.ax.tick_params(direction="out", length=10, width=3, pad=8)
    #cbar.ax.set_ylabel(data_info, ha="center", labelpad=-128)
    cbar.ax.tick_params(direction="in", length=28, width=3, pad=-75)
    cbar.ax.set_ylabel(data_info, ha="center", labelpad=0)
    
    
    sorted_columns = data.columns
    plt.xticks(range(len(sorted_columns)), sorted_columns, rotation=45, ha="right", va="top")
    plt.tick_params(axis="x", pad=-5)
    
    index = data.index
    plt.yticks(range(len(index)), index)
    
    plt.title(f"{title_info}", y=1.025)
    # plt.show()

def format_cols(mws):
    versioning = lambda x: "" if x.dataset_name.split("_")[0][-1] == "F" else x.dataset_name.split("_")[0][-1]
    multi = lambda x: "" if x.dataset_name.split("_")[1] != "FRDEBE" else " MULTI"
    
    columns = np.array([mw.string() + versioning(mw) + multi(mw) for mw in mws])
    ind_lago = np.array([type(mw) is LAGOWrapper for mw in mws])

    columns = [c.replace("_", " ") for c in columns]
    return columns, ind_lago

def plot_diff(res_1, res_2, title_info):
    indices = res_1.join(res_2, how="inner").index
    countries = np.unique(np.array([c[0] for c in indices]))
    metrics = np.unique(np.array([c[1] for c in indices]))

    res_1 = res_1.loc[indices]
    res_2 = res_2.loc[indices]

    columns = []
    columns_2 = []
    for k1 in res_1.keys():
        for k2 in res_2.keys():
            if (k1 in k2) or (k2 in k1):
                columns.append(k1)
                columns_2.append(k2)

    res_1 = res_1[columns].values
    res_2 = res_2[columns_2].values
    
    diff  = pandas.DataFrame(100 * ((res_1 - res_2) / res_1), columns=columns,
                             index=indices)
    vabs = max(abs(np.quantile(diff, 0.05)), abs(np.quantile(diff, 0.95)))
    vabs = 15
    
    if "MULTI" in columns[0]: columns = [c.split("MULTI")[0] for c in columns]
    if "2" in columns[0]: columns = [c.split("2")[0] for c in columns]
    columns = [c.split(" ")[-1] for c in columns]

    diff.columns = columns
    diff.index = [c[-1] for c in diff.index]
    
    x, y = (1, 1)
    #plt.text(x + 2.9, y + 3, "Models 2 better than  Models 1", rotation=90)
    #plt.text(x + 2.9, y + 9, "Models 1 better than  Models 2", rotation=90)    
    plot_matrix(
        diff,
        data_info="$100\\frac{" + title_info[0][1:-1] + "-" + title_info[1][1:-1] + "}{" + title_info[0][1:-1] + "}\%$",        
        title_info=f"Comparing {title_info[0]} and {title_info[1]}",
        colormap=cmap_diff_values_2(), scaler_class=None, vmin=-vabs, vmax=vabs)

    nmw = len(columns)    
    plt.xticks(range(len(columns)), [])
    for x, col in enumerate(columns):
        plt.text(x + 0.25, len(metrics) * len(countries) - 0.55, col, rotation=45, ha="right", va="top")  
    
    y0 = len(columns) * len(countries) - 1
    x0 = -0.42
    for i, country in enumerate(countries):        
        plt.text(5.1 * x0, y0 + 0.5 - ((i + 1)  * len(metrics)  + len(metrics) / 2), country)
        if i != len(countries) - 1:
            yaxes = (i + 1) / len(countries)
            plt.annotate("",
                         xy=[x0, yaxes],
                         xycoords="axes fraction",
                         xytext = [1, yaxes],                         
                         textcoords="axes fraction",                          
                         arrowprops={"arrowstyle": "-", "linestyle" : "--",
                                     "linewidth" : 5, "color" : "k"})

    i_masks = (1, )
    y0 = -0.1
    for i in range(nmw):        
        xaxes = 0.01 + (i + 1) / nmw
        if i in i_masks:
            ratio = 0.95
            plt.annotate("",
                         xy=[xaxes, 0],
                         xycoords="axes fraction",
                         xytext = [xaxes - ratio / nmw, y0],                         
                         textcoords="axes fraction",                          
                         arrowprops={"arrowstyle": "-", "linestyle" : "--",
                                     "linewidth" : 3, "color" : "k"})
            plt.annotate("",
                         xy=[xaxes - ratio / nmw, y0],
                         xycoords="axes fraction",
                         xytext = [xaxes - ratio / nmw, y0-0.1],                         
                         textcoords="axes fraction",                          
                         arrowprops={"arrowstyle": "-", "linestyle" : "--",
                                     "linewidth" : 3, "color" : "k"})             
    xindices = (1.95, )
    labels = ("SVR", )
    for xind, label in zip(xindices, labels):
        plt.text(xind, len(countries) * len(metrics) + 0.8, label, ha="center")
    return diff

def plot_scaled_metrics(results, model_wrappers, dataset, version, metrics):
    countries = [k for k in model_wrappers[version][dataset].keys()]
    nmw = len(model_wrappers[version][dataset][countries[0]])
    data = np.zeros((len(countries) * len(metrics), nmw))
    if len(countries) == 3: epf = False
    else: epf = True

    for i in range(len(countries)):
        data[i * len(metrics):(i + 1) * len(metrics), :] = results[version][dataset][i, :, :].transpose()

    columns, ind_lago = format_cols(model_wrappers[version][dataset][countries[0]])
    indices = np.concatenate([np.array([m.__name__ for m in metrics]) for c in countries])
    
    res = sort_columns(pandas.DataFrame(data, columns=columns, index=indices), ind_lago)
    col_temp = copy.deepcopy(res.columns)
    if "MULTI" in res.columns[0]: res.columns = [c.split("MULTI")[0] for c in res.columns]    
    if "2" in res.columns[0]: res.columns = [c.split("2")[0] for c in res.columns]
    res.columns = [c.split(" ")[-1] for c in res.columns]
    
    plot_matrix(res, colormap=cmap_scaled_values(), scaler_class=MinAbsScaler,
                title_info="Scaled metrics on the\nRecalibrated Forecasts",
                data_info="Fraction of the best metric", vmin=1, vmax=1.25)
    
    cols_to_display = res.columns
    res.columns = copy.deepcopy(col_temp)
    
    res.index = [(c, m.__name__) for (c, m) in itertools.product(countries, metrics)]
    plt.xticks(range(len(cols_to_display)), [])
    pad = -0.45
    for x, col in enumerate(cols_to_display):
        plt.text(x + 0.35, len(metrics) * len(countries) + pad, col, rotation=52, ha="right", va="top")    

    y0 = len(metrics) * len(countries) - 1
    x0 = -0.5
    if epf:
        pad_text = 7
        pad_line = -0.25
    else:
        pad_text = 4
        pad_line = -0.4        
    for i, country in enumerate(countries):        
        plt.text(pad_text * x0, i * len(metrics) + len(metrics) / 2, country)
        if i != len(countries) - 1:
            yaxes = (i + 1) / len(countries)
            plt.annotate("",
                         xy=[pad_line, yaxes],
                         xycoords="axes fraction",
                         xytext = [1, yaxes],                         
                         textcoords="axes fraction",                          
                         arrowprops={"arrowstyle": "-", "linestyle" : "--",
                                     "linewidth" : 5, "color" : "k"})

    if epf: i_masks = (2, 4, 8)
    else: i_masks = (2, )
    y0 = -0.1
    if epf: fraction = 1.1
    else: fraction = 0.7
    for i in range(nmw):        
        xaxes = 0.01 + (i + 1) / nmw
        if i in i_masks:
            plt.annotate("",
                         xy=[xaxes, 0],
                         xycoords="axes fraction",
                         xytext = [xaxes - fraction / nmw, y0],
                         textcoords="axes fraction",
                         arrowprops={"arrowstyle": "-", "linestyle" : "--",
                                     "linewidth" : 3, "color" : "k"})
            plt.annotate("",
                         xy=[xaxes - fraction / nmw, y0],
                         xycoords="axes fraction",
                         xytext = [xaxes - fraction / nmw, y0-0.1],   
                         textcoords="axes fraction",                          
                         arrowprops={"arrowstyle": "-", "linestyle" : "--",
                                     "linewidth" : 3, "color" : "k"})
            
            
    if epf:
        pad = 1.5
        xindices = (2.25, 5.25, 10.5)
        labels = ("SVR", "LEAR", "DNN")
    else:
        pad = 0.7
        xindices = (1.95, )
        labels = ("SVR", )        
    for xind, label in zip(xindices, labels):
        plt.text(xind, len(countries) * len(metrics) + pad, label, ha="center")
    
    return res    

def compute_metrics(predictions, model_wrappers, metrics, real_prices, naive_forecasts):
    results = {}
    versions = [k for k in predictions.keys()]
    for version in versions:
        results[version] = {}
        datasets = [k for k in predictions[version].keys()]
        for dataset in datasets:
            countries = [k for k in predictions[version][dataset].keys()]
            if len(countries) > 0:
                nmw = len(model_wrappers[version][dataset][countries[0]])
                results[version][dataset] = -np.ones((len(countries), nmw, len(metrics)))
            
                for (i, country) in enumerate(countries):
                    y_true = real_prices[dataset][country]
            
                    for (j, model_wrapper) in enumerate(
                            model_wrappers[version][dataset][country]):
                        y_pred = predictions[version][dataset][country][model_wrapper]
                        if y_pred is not None:
                            for (k, metric) in enumerate(metrics):
                                #print(version, dataset, country, model_wrapper, metric)
                                if metric == rmae:
                                    value = metric(y_true, y_pred,
                                                   naive_forecasts[dataset][country])
                                else: value = metric(y_true, y_pred)
                
                                results[version][dataset][i, j, k] = value
    
    return results

def load_forecasts(datasets, countries, models, versions, lago_params):
    # Load all predictions
    predictions = {}
    model_wrappers = {}
    for version in versions:
        predictions[version] = {}
        model_wrappers[version] = {}
        
        for dataset in datasets:            
            predictions[version][dataset] = {}
            model_wrappers[version][dataset] = {}

            if version != "3":
                for country in countries:
                    predictions[version][dataset][country] = {}   
                    model_wrappers[version][dataset][country] = []
                
                    for model in models:
                        if model != LAGOWrapper:
                            name = get_model_string(model) + "_TSCHORA"
                            model_wrappers_temp = [model(
                                name, f"EPF{version}_{country}", "")]
                        else:
                            model_wrappers_temp = []
                            for key in lago_params.keys():
                                for value in lago_params[key]:
                                    name = f"{key}_{str(value)}"
                                    model_wrappers_temp.append(model(
                                        name, f"EPF{version}_{country}", ""))
                        
                        for model_wrapper in model_wrappers_temp:
                            try:
                                all_prevs = load_prevs_mw(model_wrapper, dataset).values
                            except: all_prevs = None
                            if all_prevs is not None:
                                predictions[version][dataset][country][model_wrapper] = all_prevs
                                model_wrappers[version][dataset][country].append(
                                    model_wrapper)
                                
                    if predictions[version][dataset][country] == {}:
                        del predictions[version][dataset][country]
                        del model_wrappers[version][dataset][country]

            if version == "3":
                for (i, cc) in enumerate(("FR", "DE", "BE")):
                    predictions[version][dataset][cc] = {}
                    model_wrappers[version][dataset][cc] = []
                    for model in models:
                        name = get_model_string(model) + "_TSCHORA"
                        model_wrapper = model(name, "EPF2_FRDEBE", "")
                
                        try:
                            all_prevs = load_prevs_mw(model_wrapper, dataset).values
                        except Exception as e:
                            #print(e)
                            all_prevs = None

                        if all_prevs is not None:
                            predictions[version][dataset][cc][model_wrapper] = {}
                            ypred = all_prevs[:, 24*i:24*(i+1)]
                            predictions[version][dataset][cc][model_wrapper] = ypred
                            model_wrappers[version][dataset][cc].append(model_wrapper)

                    if predictions[version][dataset][cc] == {}:
                        del predictions[version][dataset][cc]
                        del model_wrappers[version][dataset][cc]

    return predictions, model_wrappers

def load_real_prices(countries, dataset=""):    
    real_prices = {"train" : {}, "test" : {}, "validation": {}, "test_recalibrated" : {}}
    naive_forecasts = {"train" : {}, "test" : {}, "validation": {}, "test_recalibrated" : {}}
    for country in countries:
        if country == "FRDEBE": pass
        else:
            if dataset in ("", "2", "3"): nval = 362
            else: nval = 365       
            spliter = MySplitter(nval, shuffle=False)            
            # Instantiate a default Naive Wrapper
            if dataset not in ('FRBL8', 'FRBL10', 'FRBL11'):
                labels = [f"{get_country_code(country)}_price_{i}"
                          for i in range(24)]
            else:
                labels = ["FR_price", ]
            model_wrapper = Naive("NAIVE", f"EPF{dataset}_{country}", labels)
            
            # Fill test sets
            real_prices["test"][country] = model_wrapper.load_test_dataset()[1] 
            real_prices["test_recalibrated"][country] = model_wrapper.load_test_dataset()[1]   
            
            # Need to re-split for taking the validation prices
            X, y = model_wrapper.load_train_dataset()
            ((Xtr, ytr), (Xv, yv)) = spliter(X, y)
            real_prices["validation"][country] = yv
            real_prices["train"][country] = ytr            
            
            # Also computes the naive forecasts    
            naive_forecasts["validation"][country] = model_wrapper.predict(None, Xv) 
            naive_forecasts["train"][country] = model_wrapper.predict(None, Xtr)           
            Xt, yt = model_wrapper.load_test_dataset()
            naive_forecasts["test"][country] = model_wrapper.predict(None, Xt)
            naive_forecasts["test_recalibrated"][country] = model_wrapper.predict(None, Xt)

    return real_prices, naive_forecasts
