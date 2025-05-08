import numpy as np

config = {
    "d": 10,
    "l1": 0,
    "l2": 1.0,
    "para_f": [0, 0.2, 0.4, 0.6, 0.8], ## for f
    "para_V": [[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]],
    "N": 500,
    "n": 2000,
    "m0": 1000,
    "mul_ls": np.array([1, 2, 4, 8, 16]),
    "M": 500,
    "num_quad": 64,
    "Ltype": "const",
    "Vtype": "random",
}

train_config = {
    "learning_rate": 0.001,
    "decay_rate": 0.95,
    "decay_epoch": 20,
    "epochs": 200,
}