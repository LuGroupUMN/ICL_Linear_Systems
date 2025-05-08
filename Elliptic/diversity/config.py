import numpy as np
config = {
    "d": 32,
    "l1": 0.1,
    "l2": 1.0,
    # "para_ls": [[2, 2], [2, 1], [2, 3], [2, 4], [1, 2], [3, 2], [4, 2]],
    #"para_ls": [[1, 2], [1, 3], [1, 4], [1, 5]],
    "para_ls": [[10, 20], [10, 30], [10, 40], [10, 50]],
    #"para_ls": [[100, 200], [100, 300], [100, 400], [100, 500]],
    "N": 5000,
    "n": 5000,
    "m0": 500,
    "mul_ls": np.array([1, 2, 4, 8, 16, 32]),
    "M": 500,
    "num_quad": 64,
    "Ltype": 'const3',
    "Vtype": 'piecewise',
}

train_config = {
    "learning_rate": 0.02,
    "decay_rate": 0.95,
    "decay_epoch": 200,
    "epochs": 5000,
}