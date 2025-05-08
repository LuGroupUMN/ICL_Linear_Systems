from utils.data_utils import *
from utils.train_utils import *
from config import *
import pickle
import time

device = torch.device("cuda" if torch.cuda.is_available() else
                      "mps" if torch.backends.mps.is_available() else
                      "cpu")

d, l1, l2 = config["d"], config["l1"], config["l2"]
para_f = config["para_f"]
para_V = config["para_V"]
m0, n, M, N = config["m0"], config["n"], config["M"], config["N"]
mul_ls = config["mul_ls"]
num_quad = config["num_quad"]
N_max = N + M
m_ls = m0 * mul_ls
Ltype, Vtype = config["Ltype"], config["Vtype"]


### prepare data
## train
tic =time.time()
x_train = get_f(N, d, rho=0)
A = generate_A_inv(N, d, l1=l1, l2=l2, Ltype=Ltype, Vtype=Vtype)
y_train = torch.bmm(A, x_train.unsqueeze(2)).squeeze(2)
Yn = generate_Yn(N, n, d, rho=0)
An = A @ Yn
A, An, x_train, y_train = (torch.tensor(arr, dtype=torch.float32, device=device) for arr in
                               (A, An, x_train, y_train))
toc = time.time()-tic
print('time% .3f'% toc)

data_name = f'Train_data_N_{N}_n_{n}.pkl'

data = {
    'x_train': x_train,
    'y_train': y_train,
    'A': A,
    'An': An
}

with open(data_name, 'wb') as f:
    pickle.dump(data, f)

### test data for vary V
test_data_ls = []
for para in para_V:
    tmp_a, tmp_b = para[0], para[1]
    x_test = get_f(M, d, rho=0)
    A_test = generate_A_inv(M, d, l1=l1, l2=l2, Ltype=Ltype, Vtype=Vtype, a=tmp_a, b=tmp_b)
    y_test = torch.bmm(A_test, x_test.unsqueeze(2)).squeeze(2)
    for m in m_ls:
        Ym = generate_Yn(M, m, d, type='equicorrelated', rho=0)

        Am_test = A_test @ Ym

        tmp_data = [x_test, y_test, A_test, Am_test]

        test_data_ls.append(tmp_data)

data_name = f'Test_data_var_v_M_{M}.pkl'
with open(data_name, 'wb') as f:
    pickle.dump(test_data_ls, f)

### test data for vary f
test_data_ls = []
for para in para_f:
    tmp_rho = para
    x_test = get_f(M, d, rho=tmp_rho)
    A_test = generate_A_inv(M, d, l1=l1, l2=l2, Ltype=Ltype, Vtype=Vtype)
    y_test = torch.bmm(A_test, x_test.unsqueeze(2)).squeeze(2)
    for m in m_ls:
        Ym = generate_Yn(M, m, d, type='equicorrelated', rho=tmp_rho)
        Am_test = A_test @ Ym

        tmp_data = [x_test, y_test, A_test, Am_test]

        test_data_ls.append(tmp_data)

data_name = f'Test_data_var_f_M_{M}.pkl'
with open(data_name, 'wb') as f:
    pickle.dump(test_data_ls, f)

