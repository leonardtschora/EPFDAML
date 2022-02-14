from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
import numpy as np
from statsmodels.robust import mad

class DataScaler(TransformerMixin, BaseEstimator):
    """
    Standardize the data using a Standard Scaler, then appply the arcsinh.
    """
    def __init__(self, scaling, spliter=None):
        self.scaling = scaling
        self.spliter = spliter

        if self.scaling == "BCM":
            self.scaler = BCMScaler()
        if self.scaling == "Standard":
            self.scaler = StandardScaler()
        if self.scaling == "Median":
            self.scaler = MedianScaler()
        if self.scaling == "SinMedian":
            self.scaler = SinMedianScaler()
        if self.scaling == "InvMinMax":
            self.scaler = InvMinMaxScaler()
        if self.scaling == "MinMax":
            self.scaler = MinMaxScaler()             
        if self.scaling not in ("BCM", "Standard", "", "Median", "SinMedian", "InvMinMax", "MinMax"):
            raise ValueError('Scaling parameter must be one of "BCM", "Standard", "", Median, SinMedian, InvMinMax, MinMax!')
            
    def fit(self, X, y=None):
        X = check_array(X, accept_sparse=True)
        self.n_features_ = X.shape[1]

        if self.spliter is not None:
            (X, _) = self.spliter(X)
            
        if not self.scaling == "":
            self.scaler.fit(X)
            
        self.is_fitted_ = True
        return self        
    
    def transform(self, X):
        check_is_fitted(self, 'n_features_')
        X = check_array(X, accept_sparse=True)        
        if X.shape[1] != self.n_features_:
            raise ValueError('Shape of input is different from what was seen'
                             'in `fit`')
        
        if self.scaling == "": return X
        else: return self.scaler.transform(X)

    def inverse_transform(self, X, y=None):
        check_is_fitted(self, 'n_features_')
        X = check_array(X, accept_sparse=True)        
        if X.shape[1] != self.n_features_:
            raise ValueError('Shape of input is different from what was seen'
                             'in `fit`')

        if self.scaling == "": return X
        else: return self.scaler.inverse_transform(X)


class BCMScaler(TransformerMixin, BaseEstimator):
    """
    Standardize the data using a Standard Scaler, then apply the arcsinh.
    """
    def __init__(self):
        self.scaler = StandardScaler()

    def fit(self, X, y=None):
        X = check_array(X, accept_sparse=True)
        self.n_features_ = X.shape[1]
        self.scaler.fit(X)        
        self.is_fitted_ = True
        return self        
    
    def transform(self, X):
        check_is_fitted(self, 'n_features_')
        X = check_array(X, accept_sparse=True)        
        if X.shape[1] != self.n_features_:
            raise ValueError('Shape of input is different from what was seen'
                             'in `fit`')
        
        transformed_data = self.scaler.transform(X)
        transformed_data = np.arcsinh(transformed_data)
        return transformed_data

    def inverse_transform(self, X, y=None):
        check_is_fitted(self, 'n_features_')
        X = check_array(X, accept_sparse=True)        
        if X.shape[1] != self.n_features_:
            raise ValueError('Shape of input is different from what was seen'
                             'in `fit`')
        
        transformed_data = np.sinh(X)
        transformed_data = self.scaler.inverse_transform(transformed_data)

        return transformed_data


class MedianScaler(TransformerMixin, BaseEstimator):
    """
    Standardize the data using a Median Scaler
    """
    def __init__(self, epsilon=10e-5):
        self.epsilon = epsilon

    def fit(self, X, y=None):
        X = check_array(X, accept_sparse=True)
        self.n_features_ = X.shape[1]

        self.median = np.median(X, axis=0)
        self.mad = mad(X, axis=0)
        self.mad = np.clip(self.mad, a_min=self.epsilon, a_max=None)
        
        self.is_fitted_ = True
        return self        
    
    def transform(self, X):
        check_is_fitted(self, 'n_features_')
        X = check_array(X, accept_sparse=True)        
        if X.shape[1] != self.n_features_:
            raise ValueError('Shape of input is different from what was seen'
                             'in `fit`')
        
        transformed_data = (X - self.median) / self.mad
        return transformed_data

    def inverse_transform(self, X, y=None):
        check_is_fitted(self, 'n_features_')
        X = check_array(X, accept_sparse=True)        
        if X.shape[1] != self.n_features_:
            raise ValueError('Shape of input is different from what was seen'
                             'in `fit`')

        transformed_data = (X * self.mad) + self.median        
        return transformed_data


class SinMedianScaler(TransformerMixin, BaseEstimator):
    """
    Standardize the data using a Standard Scaler, then apply the arcsinh.
    """
    def __init__(self):
        self.scaler = MedianScaler()

    def fit(self, X, y=None):
        X = check_array(X, accept_sparse=True)
        self.n_features_ = X.shape[1]
        self.scaler.fit(X)        
        self.is_fitted_ = True
        return self        
    
    def transform(self, X):
        check_is_fitted(self, 'n_features_')
        X = check_array(X, accept_sparse=True)        
        if X.shape[1] != self.n_features_:
            raise ValueError('Shape of input is different from what was seen'
                             'in `fit`')
        
        transformed_data = self.scaler.transform(X)
        transformed_data = np.arcsinh(transformed_data)
        return transformed_data

    def inverse_transform(self, X, y=None):
        check_is_fitted(self, 'n_features_')
        X = check_array(X, accept_sparse=True)        
        if X.shape[1] != self.n_features_:
            raise ValueError('Shape of input is different from what was seen'
                             'in `fit`')
        
        transformed_data = np.sinh(X)
        transformed_data = self.scaler.inverse_transform(transformed_data)

        return transformed_data

class InvMinMaxScaler(TransformerMixin, BaseEstimator):
    """
    Standardize the data using a Standard Scaler, then apply the arcsinh.
    """
    def __init__(self, epsilon=10e-5):
        self.scaler = MinMaxScaler()
        self.epsilon = epsilon

    def fit(self, X, y=None):
        X = check_array(X, accept_sparse=True)
        self.n_features_ = X.shape[1]

        transformed_data = 1 / np.clip(X, a_min=self.epsilon, a_max=None)
        self.scaler.fit(transformed_data)
        
        self.is_fitted_ = True
        return self        
    
    def transform(self, X):
        check_is_fitted(self, 'n_features_')
        X = check_array(X, accept_sparse=True)        
        if X.shape[1] != self.n_features_:
            raise ValueError('Shape of input is different from what was seen'
                             'in `fit`')
        
        transformed_data = 1 / np.clip(X, a_min=self.epsilon, a_max=None)
        transformed_data = self.scaler.transform(transformed_data)
        
        return transformed_data

    def inverse_transform(self, X, y=None):
        check_is_fitted(self, 'n_features_')
        X = check_array(X, accept_sparse=True)        
        if X.shape[1] != self.n_features_:
            raise ValueError('Shape of input is different from what was seen'
                             'in `fit`')
        
        transformed_data = self.scaler.inverse_transform(transformed_data)
        transformed_data = 1 / np.clip(transformed_data, a_min=self.epsilon, a_max=None)

        return transformed_data
    


