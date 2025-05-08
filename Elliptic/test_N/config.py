import numpy as np
factor = 1.4
config = {
    "d": 32,
    "l1": 0.1,
    "l2": 1.0,
    "N0": 500,
    "m_ls": np.array([40000, 60000, 80000]),
    "n": 50000,
    #"mul_ls": np.array([1, factor, factor**2, factor**3, factor**4, factor**5, factor**6, factor**7]),
    "mul_ls": np.array(
        [1, factor, factor ** 2, factor ** 3, factor ** 4, factor ** 5, factor ** 6, factor ** 7, factor ** 8,
         factor ** 9, factor ** 10]),
    "M": 500,
    "Ltype": 'const3',
    "Vtype": 'lognormal3',
    # "Ltype": 'const',
    # "Vtype": 'lognormal2',
}

train_config = {
    "learning_rate": 0.005,
    "decay_rate": 0.95,
    "decay_epoch": 100,
    "epochs": 5000,
}
