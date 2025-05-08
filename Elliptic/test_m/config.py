import numpy as np

config = {
    "d": 32,
    "l1": 1.0,
    "l2": 1.0,
    "N": 5000,
    #"n_ls": np.array([50, 100, 250, 500, 1000]),
    "n_ls": np.array([2000, 4000, 8000]),
    "m0": 800,
    "mul_ls": np.array([1, 2, 4, 8, 16, 32, 64, 128]),
    "M": 500,
    "Ltype": 'const3',
    "Vtype": 'lognormal3',
}


train_config = {
    "learning_rate": 0.005,
    "decay_rate": 0.95,
    "decay_epoch": 100,
    "epochs": 3000,
}