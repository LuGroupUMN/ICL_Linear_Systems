import numpy as np

config = {
    "d": 32,
    "l1": 0.1,
    "l2": 1.0,
    "para_ls": [[2, 2], [2, 1], [2, 3], [2, 4], [1, 2], [3, 2], [4, 2]],
    "alpha_1": 2,
    "beta_1": 2,
    "alpha_2": 2,
    "beta_2": 2,
    "alpha_3": 2,
    "beta_3": 2,
    "N": 50,
    "n": 20,
    "m0": 1000,
    "mul_ls": np.array([1, 2, 4, 8, 16, 32]),
    "M": 200,
    "num_quad": 64,
    "Ltype": 'lognormal',
    "Vtype": 'lognormal3',
}

train_config = {
    "learning_rate": 0.0001,
    "decay_rate": 0.95,
    "decay_epoch": 20,
    "epochs": 200,
}