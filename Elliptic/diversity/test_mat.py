from train_utils import *
from config import *
from Net import *
import pickle
from data_utils import *

d = config["d"]
n, N, M, m0 = config["n"], config["N"], config["M"], config["m0"]
mul_ls = config["mul_ls"]
N_max = N + M
para_ls = config["para_ls"]
m_ls = m0 * mul_ls

mat = generate_A_inv(1, 32, sigma=0.001, type='piecewise', dv=2, a=1, b=2)
#mat = generate_A_inv(1, 32, sigma=0.01, type='lognormal', alpha=1, beta=1)
print(mat)


#
# mat = generate_A_inv(1, 32, sigma=0.0, type='lognormal', alpha=1, beta=1)
# print(mat)
