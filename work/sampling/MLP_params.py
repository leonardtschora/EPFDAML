from tensorflow.keras import initializers
from tensorflow.keras import regularizers
import scipy.stats as stats
import numpy as np

from work.sampling.structure_sampler import structure_sampler
from work.sampling.combined_sampler import combined_sampler
from work.sampling.regularization_sampler import regularization_sampler

def make_distributions_second(n):
    return {
        "neurons_per_layer" :  combined_sampler(
        [structure_sampler(1),
         structure_sampler(2),
         structure_sampler(3),
         structure_sampler(4)],
            weights = [2, 2, 1, 1]),
            "default_activity_regularizer" : combined_sampler(
                [None,
                 regularization_sampler(types="all"),
                 regularization_sampler(types="l1"),
                 regularization_sampler(types="l2")],
                weights = [3, 1, 1, 1]),
            "dropout_rate" : stats.uniform(0, 0.5),
            "batch_norm" : stats.bernoulli(0.5),
            "batch_size" : stats.randint(10, n),
            }

def make_distributions_third(n):
    return {"neurons_per_layer" : structure_sampler(1),
            "dropout_rate" : stats.uniform(0.05, 0.4),
            "batch_size" : stats.randint(10, n)}

def make_distributions_CNN_SHUFLE(n):
    return {
        "neurons_per_layer" :  combined_sampler(
        [structure_sampler(1),
         structure_sampler(2),
         structure_sampler(3),
         structure_sampler(4)],
            weights = [2, 2, 0.75, 0.5]),
            "default_activity_regularizer" : combined_sampler(
                [None,
                 regularization_sampler(types="all"),
                 regularization_sampler(types="l1"),
                 regularization_sampler(types="l2")],
                weights = [3, 1, 1, 1]),
            "dropout_rate" : stats.uniform(0, 0.5),
            "batch_norm" : stats.bernoulli(0.5),
            "batch_size" : stats.randint(10, n),
            }



def make_distributions_single_input(n):
    return {"neurons_per_layer" :  [(2, ), (4, ), (7, ), (11, ), (16, ), (25, ),
                                    (30, ), (35, ), (40, ), (45, ), (50, ), (60, ),
                                    (70, ), (80, ), (90, ), (100, ), (125, ),
                                    (150, ), (200, ), (250, ),
                                    
                                    (4, 2), (7, 4), (11, 7), (16, 2), (16, 7),
                                    (16, 11), (25, 4), (25, 11), (25, 16), (50, 7),
                                    (50, 16), (50, 25), (50, 37),
                                    (75, 11), (75, 25), (75, 50),
                                    (100, 16), (100, 37), (100, 50),
                                    (200, 25), (200, 50), (200, 100),
                                    
                                    (7, 4, 2), (11, 7, 4), (16, 7, 2), (16, 11, 6),
                                    (25, 7, 2), (25, 11, 4), (25, 16, 11),
                                    (50, 7, 2), (50, 16, 7), (50, 25, 11),
                                    (75, 16, 7), (75, 25, 11), (75, 50, 25),
                                    (100, 25, 11), (100, 37, 16), (100, 50, 25)],
            
                 "default_activity_regularizer" : [None,
                                                   regularizers.l2(l2=0.01),
                                                   regularizers.l2(l2=0.1),
                                                   regularizers.l2(l2=0.25),
                                                   regularizers.l1(l1=0.01),
                                                   regularizers.l1(l1=0.1),
                                                   regularizers.l1(l1=0.25),
                                                   regularizers.l1_l2(l1=0.01, l2=0.01),
                                                   regularizers.l1_l2(l1=0.1, l2=0.1),
                                                   regularizers.l1_l2(l1=0.25, l2=0.25)],
            "dropout_rate" : [0.0, 0.1, 0.25, 0.5],
            "batch_norm" : [True, False],
            "batch_size" : [25, 50, 100, 250, 500, 1000, n],
            "learning_rate" : stats.loguniform(10e-6, 10e-1),
            "optimizer" : ["Adam", "SGD", "Adadelta", "Adagrad"],
            "transformer" : ["", "BCM", "Standard"],
            "scaler" : ["", "BCM", "Standard"]}


def make_distributions(n):
    return {"neurons_per_layer" :  [(25, ), (50, ), (100, ), (250, ), (400, ),
                                    (50, 25), (100, 50), (150, 75), (200, 100),
                                    (300, 150),(400, 200), (250, 50), (250, 100),
                                    (75, 50, 25), (100, 50, 25), (200, 100, 50),
                                    (400, 200, 100), (300, 150, 75), (250, 100, 50)],
            
            "default_activity_regularizer" : [None,
                                              regularizers.l2(l2=0.01),
                                              regularizers.l2(l2=0.1),
                                              regularizers.l2(l2=0.25),
                                              regularizers.l1(l1=0.01),
                                              regularizers.l1(l1=0.1),
                                              regularizers.l1(l1=0.25),
                                              regularizers.l1_l2(l1=0.01, l2=0.01),
                                              regularizers.l1_l2(l1=0.1, l2=0.1),
                                              regularizers.l1_l2(l1=0.25, l2=0.25)
            ],
            "dropout_rate" : [0.0, 0.01, 0.1, 0.25, 0.5],
            "batch_norm" : [True, False],
            "batch_size" : [25, 50, 100, 250, 500, 1000, n],
            "optimizer" : ["Adam", "SGD", "Adadelta", "Adagrad"],
            "loss_function" : [
                "MeanSquaredError",
                "MeanSquaredLogarithmicError",
                "LogCosh",
                "MeanAbsoluteError",
                "MeanAbsolutePercentageError"
            ],}

def make_distributions_base(n):
    return {"neurons_per_layer" :  [(25, ), (50, ), (100, ), (250, ), (400, ),
                                    (50, 25), (100, 50), (150, 75), (200, 100),
                                    (300, 150),(400, 200), (250, 50), (250, 100),
                                    (75, 50, 25), (100, 50, 25), (200, 100, 50),
                                    (400, 200, 100), (300, 150, 75), (250, 100, 50)],
            "dropout_rate" : [0.0, 0.1, 0.25, 0.5],
            "batch_norm" : [True, False],
            "batch_size" : [25, 50, 100, 250, 500, 1000, n],
    }

def make_distributions_CNN_2(n):
    orig = make_distributions_base(n)
    orig.update({
        "conv_activatiion" : ["relu", "sigmoid"],
        "filter_size" : [(4, 2), (8, 4), (12, 6), (18, 9), (24, 12), (36, 18), (42, 21),
                         (56, 28), (64, 32), (80, 40), (100, 50),
                         (4, 3), (8, 6), (12, 9), (18, 12), (24, 16), (36, 24), (42, 31),
                         (56, 38), (64, 44), (80, 65), (100, 75),
                         (8, 3), (12, 4), (18, 6), (24, 9), (36, 12), (42, 15),
                         (56, 20), (64, 24), (80, 30), (100, 35)
        ],
        "dilation_rate" : [((1, 1), (1, 1), ), ((2, 2), (2, 2), ), ((3, 3), (3, 3), ),
                           ((3, 3), (1, 1), ), ((2, 2), (1, 1), ), ((4, 4), (2, 2), )],

        "kernel_size" : [((5, 5), (5, 5)), ((3, 3), (3, 3)), ((7, 7), (7, 7)),
                         ((9, 9), (9, 9)), ((2, 2), (2, 2))],
        "pool_size" : [((2, 2), (2, 2)), ((3, 3), (3, 3)), ((4, 4), (4, 4)),
                       ((3, 3), (2, 2)), ((4, 4), (2, 2))],
        "strides" : [(2, 2), (3, 3), (1, 1), (4, 4)],
    })
    return orig


def make_distributions_CNN_1(n):
    orig = make_distributions_base(n)
    orig.update({
        "conv_activatiion" : ["relu", "sigmoid"],
        "filter_size" : [(4, ), (8, ), (12, ), (18, ), (24, ), (36, ), (42, ),
                         (56, ), (64, ), (80, ), (100, ),
                         (3, ), (6, ), (9, ), (16, ), (31, ),
                         (38, ), (44, ), (65, ), (75, )],
        "dilation_rate" : [((1, 1), ), ((2, 2), ), ((3, 3), ), ((4, 4),  )],

        "kernel_size" : [((5, 5), ), ((3, 3), ), ((7, 7), ),
                         ((9, 9), ), ((2, 2), )],
        "pool_size" : [((2, 2), ), ((3, 3), ), ((4, 4), )],
        "strides" : [(2, ), (3, ), (1, ), (4, )],
    })
    return orig

def make_distributions_CNN_3(n):
    orig = make_distributions_base(n)
    orig.update({
        "conv_activatiion" : ["relu", "sigmoid"],
        "filter_size" : [(8, 4, 2), (12, 6, 3), (18, 9, 5), (24, 12, 6), (36, 18, 9),
                         (42, 21, 10), (56, 28, 14), (64, 32, 16), (80, 40, 20),
                         (100, 50, 25),
                         (8, 6, 4), (12, 9, 6), (18, 12, 9), (24, 16, 12),
                         (36, 24, 18), (42, 31, 21), (56, 38, 28), (64, 44, 32),
                         (80, 65, 40), (100, 75, 50),
                         (18, 6, 2), (24, 9, 3), (36, 12, 4), (42, 15, 6),
                         (56, 20, 8), (64, 24, 10), (80, 30, 12), (100, 35, 15)
        ],
        "dilation_rate" : [((1, 1), (1, 1), (1, 1),),
                           ((2, 2), (2, 2), (2, 2),),
                           ((3, 3), (3, 3), (3, 3),),
                           
                           ((3, 3), (2, 2), (1, 1),),
                           ((4, 4), (3, 3), (2, 2),),
                           ((5, 5), (3, 3), (1, 1),)],
        
        "kernel_size" : [((2, 2), (2, 2), (2, 2),),
                         ((3, 3), (3, 3), (3, 3),),
                         ((5, 5), (5, 5), (5, 5),),
                         ((7, 7), (7, 7), (7, 7),),
                         ((9, 9), (9, 9), (9, 9),)],
        "pool_size" : [((2, 2), (2, 2), (2, 2),),
                       ((3, 3), (3, 3), (3, 3),),
                       ((4, 4), (4, 4), (4, 4),),
                       ((4, 4), (3, 3), (2, 2),),
                       ((5, 5), (4, 4), (3, 3),)],
        "strides" : [(1, 1, 1), (2, 2, 2), (3, 3, 3), (4, 4, 4)],
    })
    return orig
