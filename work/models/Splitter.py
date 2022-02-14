from sklearn.model_selection import PredefinedSplit
import numpy as np
from numpy.random import default_rng

SPLITER_SEED = 90125

class MySplitter(object):
    def __init__(self, validation_ratio, shuffle=False):
        self.validation_ratio = validation_ratio
        self.shuffle = shuffle

    def get_splitter(self, X):
        n = X.shape[0]        
        mask = -np.ones(n)

        # If n is below 1, it is used as a validation ratio, otherwise as the
        # number of validation_samples.
        if self.validation_ratio < 1:
            ntr = int((1 - self.validation_ratio) * n)
        else:
            ntr = n - self.validation_ratio
            
        mask[ntr:] = 0
        if self.shuffle:
            rng = default_rng(SPLITER_SEED)
            mask = rng.permutation(mask)
            
        cv = PredefinedSplit(mask)
        return cv
    
    def __call__(self, X, y=None):
        cv = self.get_splitter(X)
        
        trind, vind = [ind for ind in cv.split(X, y)][0]
        Xtr = X[trind]
        Xv = X[vind]
        
        if y is None:
            return (Xtr, Xv)

        ytr = y[trind]
        yv = y[vind]

        return ((Xtr, ytr), (Xv, yv))

    def __str__(self):
        return f"MySplitter_{str(self.validation_ratio)}_str{self.shuffle}"
        
