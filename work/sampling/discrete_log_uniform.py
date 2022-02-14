from tensorflow.keras import regularizers
import scipy.stats as stats
from scipy.stats import rv_continuous

class discrete_loguniform(rv_continuous):
    """
    Simulates a discrete log-uniform sampling.
    Samples from a continuous log-uniform distribution and then round the result.
    """
    def __init__(self, min_, max_):
        rv_continuous.__init__(self)
        self.distribution = stats.loguniform(min_, max_)

    def rvs(self, size=1, random_state=None):
        sampled = self.distribution.rvs(size=size, random_state=random_state)
        rounded_samples = [int(sample) for sample in sampled]

        if size == 1:
            rounded_samples = rounded_samples[0]
            
        return rounded_samples
        
            
