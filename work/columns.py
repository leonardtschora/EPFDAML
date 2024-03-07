import pandas, numpy as np
import work.TSCHORA_results_utils as tru

def labels_and_pos(model_wrapper, version, country, for_plot=True):
    """
    2 fonctions : 
    1) retourner un ordre de colonnes pour trier le dataset comme on le souhaite
    après sa lecture sur le disque. C'est utile pour arranger les colonnes du
    dataframe pour entraîner un CNN par exemple : pour que les filtres soient
    efficaces, il faut que les features soient bien regroupées. On spécifiera dans
    ce cas for_plot = False et l'on ne s'intéressera qu'au premier éleḿent de 
    la liste des sorties de cette fonction

    2) Trier les colonnes du dataset pour calculer/plotter les shap values. Les
    shap values ont bien plus de sens et sont plus lisibles si on les regroupe par
    features, il est donc nécéssaire de bien trier les colonnes du dataset.    
    Dans ce cas, on laisse for_plot = True, et on s'intéresse aux 5 valeurs de la 
    liste des arguments de retour:
    
    feature_names_ordered
    feature_labels
    label_pos
    sort_indices
    lags
    """    
    if for_plot:
        sort_func = lambda x: np.sort(x)[::-1]
        order_concat = lambda x, y: x + y
    else:
        sort_func = lambda x: np.sort(x)
        order_concat = lambda x, y: y + x
        
    data = pandas.read_csv(model_wrapper.train_dataset_path()).drop(
        columns=model_wrapper.label + ["period_start_date"])
    feature_names = data.keys()
    cc = tru.get_country_code(country)
     
    # Set order for better display
    if version == "":
        prfx, lags, feature_labels, feature_names_ordered, i_init = label_and_pos_v1(cc, country, sort_func, order_concat)
    elif ((version == "2") and (country != "FRDEBE")):
        prfx, lags, feature_labels, feature_names_ordered, i_init = label_and_pos_v2(cc, country, sort_func, order_concat)
    # TODO : add and (country=='FRBE')
    elif ((version == "2") and (country == "FRDEBE")):
        prfx, lags, feature_labels, feature_names_ordered, i_init = label_and_pos_v3(cc, country, sort_func, order_concat)
    elif ((version == "3") and (country != "FRDEBE")):
        prfx, lags, feature_labels, feature_names_ordered, i_init = label_and_pos_v4(cc, country, sort_func, order_concat)
    
    else:
        return feature_names, None

    label_pos = [0, 8]
    i = i_init
    for prefix in prfx:
        for lag in lags[prefix]:
            feature_labels.append(f"{prefix}{lag}")
            label_pos.append(i)
            for hour in range(0, 24):
                i += 1
                feature_names_ordered.append(f"{prefix}_{hour}{lag}")

    sort_indices = np.zeros(len(feature_names), int)
    for j, feature_name in enumerate(feature_names_ordered):
        sort_indices[j] = np.where(feature_names == feature_name)[0][0]

    return feature_names_ordered, feature_labels, label_pos, sort_indices, lags


def label_and_pos_v1(cc, country, sort_func, order_concat):
    prefixes = {"FR" : ["FR_System load forecast", "FR_Generation forecast"],
                "DE" : ["DE_Ampirion Load Forecast", "DE_PV+Wind Forecast"],
                "BE" : ["BE_System load forecast", "BE_Generation forecast"],
                "NP" : ["NOR_Grid load forecast", "NOR_Wind power forecast"],
                "PJM" : ["US_Zonal COMED load foecast", "US_System load forecast"]}
    
    date_f = f"{cc}_day_of_week"        
    prfx = [prefix for prefix in prefixes[country] + [f"{cc}_price"]]
    lags = {}
    for k in prfx:
        if k == f"{cc}_price":
            lags[k] = [f"_past_{i}" for i in sort_func((7, 3, 2, 1))]
        else:
            lags[k] = order_concat([f"_past_{i}" for i in sort_func((7, 1))], [""])
            
    feature_names_ordered = [date_f]
    feature_labels = [date_f]
    i_init = 1
    return prfx, lags, feature_labels, feature_names_ordered, i_init

def label_and_pos_v2(cc, country, sort_func, order_concat):
        prfx = ["FR_energy", "FR_Generation forecast", "FR_price",
                "DE_price", "DE_Ampirion Load Forecast", "DE_PV+Wind Forecast",
                "BE_price", "NL_price"]
        c1, c2 = [i for i in ["FR", "DE", "BE"] if i != country] 
        lags = {
            "FR_energy" : order_concat([f"_past_{i}" for i in sort_func((7, 1))], [""]),

            "FR_Generation forecast" : order_concat([f"_past_{i}" for i in sort_func((7, 1))], [""]),

            f"{country}_price" : [f"_past_{i}" for i in sort_func((7, 3, 2, 1))],

            "DE_Ampirion Load Forecast" : order_concat([f"_past_{i}" for i in sort_func((7, 1))], [""]),

            "DE_PV+Wind Forecast" : order_concat([f"_past_{i}" for i in sort_func((7, 1))], [""]),
            f"{c1}_price" : [f"_past_{i}" for i in sort_func((7, 1))],
            f"{c2}_price" : [f"_past_{i}" for i in sort_func((7, 1))],
            "NL_price" : [f"_past_{i}" for i in sort_func((7, 1))]}
        
        if country == "DE" or country == "FR":
            lags["CH_price"] = order_concat(
                [f"_past_{i}" for i in sort_func((7, 1))], [""])
            prfx.append("CH_price")
        if country == "FR":
            lags["ES_price"] = [f"_past_{i}" for i in sort_func((7, 1))]
            prfx.append("ES_price")
            
        feature_names_ordered = [
            f'{cc}_day_1', f'{cc}_day_2', f'{cc}_day_of_week_1',            
            f'{cc}_day_of_week_2', f'{cc}_week_1', f'{cc}_week_2',
            f'{cc}_month_1', f'{cc}_month_2']
        feature_labels = ["DATE"]
        i_init = 8
        return prfx, lags, feature_labels, feature_names_ordered, i_init

def label_and_pos_v3(cc, country, sort_func, order_concat):
        prfx = ["FR_energy", "FR_Generation forecast", "FR_price",
                "DE_price", "DE_Ampirion Load Forecast", "DE_PV+Wind Forecast",
                "BE_price", "NL_price", "CH_price", "ES_price"]
        lags = {
            "FR_energy" : order_concat(
                [f"_past_{i}" for i in sort_func((7, 1))], [""]),
            "FR_Generation forecast" : order_concat(
                [f"_past_{i}" for i in sort_func((7, 1))], [""]),
            "DE_Ampirion Load Forecast" : order_concat(
                [f"_past_{i}" for i in sort_func((7, 1))], [""]),
            "DE_PV+Wind Forecast" : order_concat(
                [f"_past_{i}" for i in sort_func((7, 1))], [""]),
            "FR_price" : [f"_past_{i}" for i in sort_func((7, 3, 2, 1))],
            "DE_price" : [f"_past_{i}" for i in sort_func((7, 3, 2, 1))],
            "BE_price" : [f"_past_{i}" for i in sort_func((7, 3, 2, 1))],
            "NL_price" : [f"_past_{i}" for i in sort_func((7, 1))],
            "CH_price" : order_concat(
                [f"_past_{i}" for i in sort_func((7, 1))], [""]),
            "ES_price" : [f"_past_{i}" for i in sort_func((7, 1))]}
        cc = "FR"
        feature_names_ordered = [
            f'{cc}_day_1', f'{cc}_day_2', f'{cc}_day_of_week_1',
            f'{cc}_day_of_week_2', f'{cc}_week_1', f'{cc}_week_2',
            f'{cc}_month_1', f'{cc}_month_2']
        feature_labels = ["DATE"]
        i_init = 8
        return prfx, lags, feature_labels, feature_names_ordered, i_init

def label_and_pos_v4(cc, country, sort_func, order_concat):
    prfx = ["FR_consumption", "FR_production", "FR_price",
            "DE_consumption", "DE_production", "DE_price",
            "BE_consumption", "BE_renewables_production",
            "BE_price", "NL_price"]
    c1, c2 = [i for i in ["FR", "DE", "BE"] if i != country] 
    lags = {
        "FR_consumption" : order_concat(
            [f"_past_{i}" for i in sort_func((7, 1))], [""]),
        "FR_production" : order_concat(
            [f"_past_{i}" for i in sort_func((7, 1))], [""]),
        f"{country}_price" : [f"_past_{i}" for i in sort_func((7, 3, 2, 1))],
        "DE_consumption" : order_concat(
            [f"_past_{i}" for i in sort_func((7, 1))], [""]),
        "DE_production" : order_concat(
            [f"_past_{i}" for i in sort_func((7, 1))], [""]),
        "BE_consumption" : order_concat(
            [f"_past_{i}" for i in sort_func((7, 1))], [""]),
        "BE_renewables_production" : order_concat(
            [f"_past_{i}" for i in sort_func((7, 1))], [""]),            
        f"{c1}_price" : [f"_past_{i}" for i in sort_func((7, 1))],
        f"{c2}_price" : [f"_past_{i}" for i in sort_func((7, 1))],
        "NL_price" : [f"_past_{i}" for i in sort_func((7, 1))]}
    
    if country == "DE" or country == "FR":
        lags["CH_price"] = order_concat(
            [f"_past_{i}" for i in sort_func((7, 1))], [""])
        prfx.append("CH_price")
    if country == "FR":
        lags["ES_price"] = [f"_past_{i}" for i in sort_func((7, 1))]
        prfx.append("ES_price")
        
    feature_names_ordered = [
        f'{cc}_day_1', f'{cc}_day_2', f'{cc}_day_of_week_1',            
        f'{cc}_day_of_week_2', f'{cc}_week_1', f'{cc}_week_2',
        f'{cc}_month_1', f'{cc}_month_2', "FR_ShiftedGazPrice"]
    feature_labels = ["DATE", "GAZ PRICES"]
    i_init = 9
    return prfx, lags, feature_labels, feature_names_ordered, i_init
