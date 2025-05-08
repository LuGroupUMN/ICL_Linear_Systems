import numpy as np
config = {
    "d": 10,
    "l1": 0,
    "l2": 1,
    #"para_ls": [[2, 3], [3, 4], [4, 5], [5, 6]],
    "para_ls": [[1, 2], [1, 3], [1, 4], [1, 5]],
    #"para_ls": [[1, 0.6], [1, 0.7], [1, 0.8]],
    "N": 100,
    "n": 5000,
    "m0": 500,
    "mul_ls": np.array([1, 2, 4, 8, 16, 32]),
    "M": 500,
    "num_quad": 64,
    "Ltype": 'const',
    "Vtype": 'piecewise',
}

train_config = {
    "learning_rate": 0.001,
    "decay_rate": 0.95,
    "decay_epoch": 20,
    "epochs": 200,
}