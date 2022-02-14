import scipy.stats as stats
from scipy.stats import rv_continuous
import numpy as np

class combined_sampler(rv_continuous):
    """
    Sample form a weighted combination of distributions
    """
    def __init__(self, others, weights=1):
        rv_continuous.__init__(self)
        self.others = others
        self.weights = weights

        if type(self.weights) == list:
            self.weights = [v/sum(self.weights) for v in self.weights]
        else:
            self.weights = [1/len(self.others) for o in self.others]

        self.chooser = np.random.choice

    def rvs(self, size=1, random_state=None):
        choices = self.chooser(self.others, size=size, p=self.weights)        
        samples = []
        for choice in choices:
            try:
                sampled = choice.rvs(1, random_state=random_state)[0]
            except:
                try:
                    sampled = choice.rvs(1, random_state=random_state)
                except:
                    sampled = choice
            samples.append(sampled)                
                
        if size == 1:
            samples = samples[0]
            
        return samples
