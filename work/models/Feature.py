from work.models.model_wrapper import *
import work.model_utils as mu
import datetime

class Feature(ModelWrapper):
    def __init__(self, prefix, dataset_name, label, col_idxs=None):
        ModelWrapper.__init__(self, prefix, dataset_name, label)

        if col_idxs is None: col_idxs = [l + "_past_1" for l in label]
        
        train = pandas.read_csv(self.train_dataset_path())
        train_keys = [k for k in train.keys()
                      if k not in label + ["period_start_date"]]
        idxs = []
        for col_idx in col_idxs:
            idxs.append([i for i, e in enumerate(train_keys) if e == col_idx][0])
        
        self.col_idxs = idxs

    def params(self):
        return {}
    
    def make(self, ptemp):
        return None

    def predict(self, regr, X):
        return X[:, self.col_idxs]

class Naive(ModelWrapper):
    """
    Naive forecaster which predicts the price of last day for any week days and 
    the price of last week for the week end days.
    """
    def __init__(self, prefix, dataset_name, label):
        ModelWrapper.__init__(self, prefix, dataset_name, label)
        train = pandas.read_csv(self.train_dataset_path())
        train_keys = [k for k in train.keys() if k not in label]

        day_col_idxs = []
        try:
            for col_idx in [l + "_past_1" for l in label]:
                day_col_idxs.append(
                    [i for i, e in enumerate(train_keys) if e == col_idx][0])
        except: pass
        
        week_col_idxs = []
        try:        
            for col_idx in [l + "_past_7" for l in label]:
                week_col_idxs.append(
                    [i for i, e in enumerate(train_keys) if e == col_idx][0])
        except: pass       
        
        self.day_col_idxs = day_col_idxs
        self.week_col_idxs = week_col_idxs

    def load_dataset(self, path):
        """
        Create a day of the week column
        """
        dataset = pandas.read_csv(path)
        labels = dataset[np.array(self.label)]
        dataset.drop(columns=self.label, inplace=True)

        dates = dataset.period_start_date
        dayofweek = [datetime.datetime.strptime(d, "%Y-%m-%d").weekday() for d in dates]
        dataset["DayOfWeek"] = dayofweek

        X = dataset.values 
        y = labels.values        
        return X, y
        
    def params(self):
        return {}
    
    def make(self, ptemp):
        return None

    def predict(self, regr, X):
        we_indices = (X[:, -1] == 6) + (X[:, -1] == 5)        
        we_indices = np.array([we_indices for i in range(len(self.label))]).transpose()
        return np.where(we_indices, X[:, self.week_col_idxs], X[:, self.day_col_idxs])
    
