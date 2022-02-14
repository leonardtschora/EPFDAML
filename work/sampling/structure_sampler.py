import scipy.stats as stats
from scipy.stats import rv_continuous
import numpy as np
class structure_sampler(rv_continuous):
    """
    Sample a tuple of element. Each element is drawn from a separated 
    randint dsitribution between min_ and max_. 

    The tuple is then sorted if specified.
    """
    def __init__(self, nlayers, min_=10, max_=500, sort=True):
        rv_continuous.__init__(self)
        self.nlayers = nlayers
        self.max_ = max_
        self.min_ = min_
        self.sort = sort

        distributions = []
        for i in range(self.nlayers):
            distributions.append(stats.randint(self.min_, self.max_))

        self.distributions = distributions
            
    
    def rvs(self, size=1, random_state=None):        
        all_samples = []
        for i in range(size):
            sampled = []
            for d in self.distributions:
                sampled.append(d.rvs(random_state=random_state))
                
            if self.sort:
                sampled.sort(reverse=True)

            all_samples.append(tuple(sampled))
        return all_samples

    
class double_structure_sampler(rv_continuous):
    """
    Same as structure_sampler but Each element of the tuple is a tuple of 2 values.
    If different is True, then the 2 elements of the tuple will be different.
    Otherwise, they stay the same.
    """    
    def __init__(self, nlayers, min_=(10, 10), max_=(500, 500), sort=True, different=True):
        rv_continuous.__init__(self)
        self.nlayers = nlayers
        self.different = different       
        self.distributions = [
            structure_sampler(nlayers, min_=min_[0], max_=max_[0], sort=sort),
            structure_sampler(nlayers, min_=min_[1], max_=max_[1], sort=sort)]
                        
    def rvs(self, size=1, random_state=None):        
        all_samples = []
        for i in range(size):
            sampled = self.distributions[0].rvs(random_state=random_state)[0]
            if self.different:
                second_sampled = self.distributions[1].rvs(random_state=random_state)[0]
            else:
                second_sampled = sampled

            sampled = (sampled, second_sampled)
            sampled = tuple([tuple(a) for a in np.array(sampled).transpose()])
            all_samples.append(sampled)
            
        return all_samples    
