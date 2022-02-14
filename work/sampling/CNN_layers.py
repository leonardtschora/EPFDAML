import scipy.stats as stats
from scipy.stats import rv_continuous
from work.sampling.structure_sampler import double_structure_sampler, structure_sampler
import copy

def flatten_dict(t, first=False):
    """
    Convert an array of dict into a dict of arrays
    """
    res = {}
    for d in t:
        for k in d.keys():
            if k not in res.keys():
                res[k] = []
            if not first: res[k].append(tuple(copy.deepcopy(d[k])))
            else: res[k].append(copy.deepcopy(d[k][0]))
    return res

class CNN_structure_sampler(rv_continuous):
    """
    Sampler for a CNN structure.
    A CNN structure has the following shape :
    (CONV2D * ni + POOLING * pi) * nblocks + Fully Connected Part.

    This samples does not take care of the Fully Connected Part. 
    It starts by selecting a number of Convolutional "blocks" composed by several
    stacked CONV2D layers and a pooling layer... or not.

    For each of the m blocks, a number of CONV2D layers ni is sampled, along with 
    a boolean indicating wether or not this block will have a pooling layer pi.

    For each CONV2D and Pooling layer, a configuration is then sampled, using 
    respectively a conv2d_sampler and a pooling_sampler.
    """
    def __init__(self, nblocks, W, H, P, max_filters=25, min_filters=3,
                 max_kernel=(11, 25), min_=1, max_=4):
        rv_continuous.__init__(self)
        self.max_filters = max_filters
        self.max_kernel = max_kernel
        self.min_filters = min_filters
        
        self.nlayers_sampler = structure_sampler(nblocks, min_=1, max_=4)
        self.pool_sampler = pool_sampler(W, H, P, nblocks, min_=min_, max_=max_)

    def rvs(self, size=1, random_state=None):
        all_samples = []        
        for i in range(size):
            nlayers = self.nlayers_sampler.rvs(size=1, random_state=random_state)[0]
            self.conv2d_samplers = [conv2d_sampler(
                nlayer, max_filters=self.max_filters, min_filters=self.min_filters,
                max_kernel=self.max_kernel) for nlayer in nlayers]
            conv2ds = [d.rvs(size=1, random_state=random_state)
                       for d in self.conv2d_samplers]
            
            all_samples.append(flatten_dict(conv2ds, first=True))

        pool = self.pool_sampler.rvs(size=size, random_state=None)       

        res = {}
        res.update(pool)
        
        structure = flatten_dict(all_samples)
        res.update(structure)        
        if size == 1:
            for k in res.keys():
                res[k] = res[k][0]
        return res

    
class conv2d_sampler(rv_continuous):
    """
    Sampler for sampling conv2d layer configuration for a convolutional block with 
    nlayer layers.
    Each layer consists in a 2-element tuple storing the kernel size, a single int 
    being the filter size, and a 2-element tuple for the dilation rates.
    """
    def __init__(self, nlayers, max_filters=50, min_filters=3, max_kernel=(11, 25),
                 max_dilation=(5, 5)):
        rv_continuous.__init__(self)        
        self.filter_distribution = structure_sampler(nlayers, min_=min_filters,
                                                     max_=max_filters, sort=False)
        self.kernel_distribution = double_structure_sampler(
            nlayers, min_=(2, 2), max_=max_kernel, different=True, sort=False)
        self.dilation_distribution = double_structure_sampler(
            nlayers, min_=(1, 1), max_=max_dilation, different=True, sort=False)

    def rvs(self, size=1, random_state=None):
        filters = self.filter_distribution.rvs(size=size, random_state=random_state)
        kernels = self.kernel_distribution.rvs(size=size, random_state=random_state)
        dilations = self.dilation_distribution.rvs(size=size, random_state=random_state)

        res = {"kernel_size" : kernels, "filter_size" : filters,
               "dilation_rate" : dilations}
        return res
        

class pool_sampler(rv_continuous):
    """
    Samples the pooler layer configurations for a CNN with N convolutional blocks.
    The sampler first decides wether or not there will be a pooling layer by using a
    bernouilli distribution of paramter P.
    
    If the block has a pooling layer, then 2 2-element tuples are sampled : the 
    pool size and the strides. 

    Note that the strides are reduced if they leed to an impossible configuration.
    
    """
    def __init__(self, W, H, P, nlayers, min_=1, max_=4):
        rv_continuous.__init__(self)
        self.W = W
        self.H = H
        self.min_ = min_
        self.max_ = max_
        self.P = P
        self.nlayers = nlayers

        # Distirbutions to decide wether or not to include a pooling operation
        self.P_dist = stats.bernoulli(P)

        # Distirbutions for sampling S and F
        self.S_dist = [stats.randint(self.min_, self.max_),
                       stats.randint(self.min_, self.max_)]
        self.F_dist = [stats.randint(self.min_, self.max_),
                       stats.randint(self.min_, self.max_)]

    def check_pool(self, value, stride, pool_size):
        return not ((value - pool_size) % stride)
    
    def rvs(self, size=1, random_state=None):        
        all_samples = {"pool_size" : [], "strides" : []}
        for i in range(size):
            sampled = {"pool_size" : [], "strides" : []}
            W = self.W
            H = self.H            
            for l in range(self.nlayers):
                pooling = self.P_dist.rvs(random_state=random_state)
                if not pooling:
                    sampled["pool_size"].append(None)
                    sampled["strides"].append(None)
                else:
                    strides = [d.rvs(random_state=random_state) for d in self.S_dist]
                    pool_sizes = [d.rvs(random_state=random_state) for d in self.S_dist]

                    # Reduce the stide value of the corresponding dim until we found
                    # a valid configuration
                    while not(self.check_pool(W, strides[0], pool_sizes[0])
                              and self.check_pool(H, strides[1], pool_sizes[1])):
                        if not self.check_pool(W, strides[0], pool_sizes[0]):
                            strides[0] -= 1
                        if not self.check_pool(H, strides[1], pool_sizes[1]):
                            strides[1] -= 1
                    sampled["pool_size"].append(tuple(pool_sizes))
                    sampled["strides"].append(tuple(strides))
                    W = 1 + (W - pool_sizes[0]) / strides[0]
                    H = 1 + (H - pool_sizes[1]) / strides[1]
                    
            all_samples["pool_size"].append(tuple(sampled["pool_size"]))
            all_samples["strides"].append(tuple(sampled["strides"]))
                
        return all_samples

    
