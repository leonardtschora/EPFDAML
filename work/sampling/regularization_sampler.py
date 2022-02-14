from tensorflow.keras import regularizers
import scipy.stats as stats
from scipy.stats import rv_continuous

class regularization_sampler(rv_continuous):
    """
    Sample regularization configurations.
    This corresponds to sampling a float between alpha max and alpha min using the
    specified scale (either lin for uniform distribution of log for loguniform distribution).

    Also samples the nature of the regularization : l1, l2 or both.
    """
    def __init__(self, alpha_max=0.5, alpha_min=10e-5, alpha_scale="lin", types="all"):
        rv_continuous.__init__(self)
        self.alpha_max = alpha_max
        self.alpha_min = alpha_min
        self.alpha_scale = alpha_scale
        self.types = types

        distributions = []
        
        if self.alpha_scale == "lin": distribution = stats.uniform
        if self.alpha_scale == "log": distribution = stats.loguniform

        ndistributions = 1        
        if types == "all":
            regularization = regularizers.l1_l2
            ndistributions = 2
        if types == "l1":
            regularization = regularizers.l1
        if types == "l2":
            regularization = regularizers.l2        
            
        for i in range(ndistributions):
            distributions.append(distribution(self.alpha_min, self.alpha_max))

        self.distributions = distributions
        self.regularization = regularization
            
    
    def rvs(self, size=1, random_state=None):
        all_samples = []
        for i in range(size):
            sampled_alphas = [d.rvs(random_state=random_state)
                              for d in self.distributions]
            
            if len(sampled_alphas) == 1:
                sampled = self.regularization(sampled_alphas[0])
            if len(sampled_alphas) == 2:
                sampled = self.regularization(sampled_alphas[0], sampled_alphas[1])
                
            all_samples.append(sampled)
            
        return all_samples
        
