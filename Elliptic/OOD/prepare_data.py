from utils.data_utils import *
from utils.train_utils import *
from config import *
import pickle
import time

device = torch.device("cuda" if torch.cuda.is_available() else
                      "mps" if torch.backends.mps.is_available() else
                      "cpu")

d, l1, l2 = config["d"], config["l1"], config["l2"]
para_ls = config["para_ls"]
alpha_1, beta_1, alpha_2, beta_2, alpha_3, beta_3 = config["alpha_1"], config["beta_1"], config["alpha_2"], config["beta_2"], config["alpha_3"], config["beta_3"]
m0, n, M, N = config["m0"], config["n"], config["M"], config["N"]
mul_ls = config["mul_ls"]
num_quad = config["num_quad"]
Ltype, Vtype = config["Ltype"], config["Vtype"]
N_max = N + M
m_ls = m0 * mul_ls


### prepare data
## train
tic =time.time()
x_train = get_f(N, d, alpha_3, beta_3)
A = generate_A_inv(N, d, l1=l1, l2=l2, Ltype=Ltype, Vtype=Vtype, alpha1=alpha_1, beta1=beta_1, alpha2=alpha_2, beta2=beta_2)
y_train = torch.bmm(A, x_train.unsqueeze(2)).squeeze(2)
Yn = generate_Yn(N, n, d, type='lognormal', alpha3=alpha_3, beta3=beta_3)
An = A @ Yn
A, An, x_train, y_train = (torch.tensor(arr, dtype=torch.float32, device=device) for arr in
                               (A, An, x_train, y_train))
toc = time.time()-tic
print('time% .3f'% toc)

data_name = f'Train_data_N_{N}_n_{n}_alp1_{alpha_1}_beta1_{beta_1}_alp2_{alpha_2}_beta2_{beta_2}_alp3_{alpha_3}_beta3_{beta_3}.pkl'

data = {
    'x_train': x_train,
    'y_train': y_train,
    'A': A,
    'An': An
}

with open(data_name, 'wb') as f:
    pickle.dump(data, f)

### test data for vary a
test_data_ls = []
for para in para_ls:
    tmp_alpha_1, tmp_beta_1 = para[0], para[1]
    x_test = get_f(M, d, alpha_3, beta_3)
    A_test = generate_A_inv(M, d, l1=l1, l2=l2, Ltype=Ltype, Vtype=Vtype, alpha1=tmp_alpha_1, beta1=tmp_beta_1, alpha2=alpha_2, beta2=beta_2)
    y_test = torch.bmm(A_test, x_test.unsqueeze(2)).squeeze(2)
    for m in m_ls:
        Ym = generate_Ym_small(M, m, d, alpha_3, beta_3)

        Am_test = A_test @ Ym

        #tmp_data = [x_test, y_test, A_test, Am_test]
        tmp_data = [x_test.detach().cpu().numpy(), y_test.detach().cpu().numpy(), A_test.detach().cpu().numpy(), Am_test.detach().cpu().numpy()]

        test_data_ls.append(tmp_data)

data_name = f'Test_data_var_a_M_{M}_alp1_{alpha_1}_beta1_{beta_1}_alp2_{alpha_2}_beta2_{beta_2}_alp3_{alpha_3}_beta3_{beta_3}.pkl'
with open(data_name, 'wb') as f:
    pickle.dump(test_data_ls, f)

### test data for vary V
test_data_ls = []
for para in para_ls:
    tmp_alpha_2, tmp_beta_2 = para[0], para[1]
    x_test = get_f(M, d, alpha_3, beta_3)
    A_test = generate_A_inv(M, d, l1=l1, l2=l2, Ltype=Ltype, Vtype=Vtype, alpha1=alpha_1, beta1=beta_1, alpha2=tmp_alpha_2, beta2=tmp_beta_2)
    y_test = torch.bmm(A_test, x_test.unsqueeze(2)).squeeze(2)
    for m in m_ls:
        Ym = generate_Ym_small(M, m, d, alpha_3, beta_3)
        Am_test = A_test @ Ym

        #tmp_data = [x_test, y_test, A_test, Am_test]
        tmp_data = [x_test.detach().cpu().numpy(), y_test.detach().cpu().numpy(), A_test.detach().cpu().numpy(),
                    Am_test.detach().cpu().numpy()]

        test_data_ls.append(tmp_data)

data_name = f'Test_data_var_V_M_{M}_alp1_{alpha_1}_beta1_{beta_1}_alp2_{alpha_2}_beta2_{beta_2}_alp3_{alpha_3}_beta3_{beta_3}.pkl'
with open(data_name, 'wb') as f:
    pickle.dump(test_data_ls, f)


### test data for vary f
test_data_ls = []
for para in para_ls:
    tmp_alpha_3, tmp_beta_3 = para[0], para[1]
    x_test = get_f(M, d, tmp_alpha_3, tmp_beta_3)
    A_test = generate_A_inv(M, d, l1=l1, l2=l2, Ltype=Ltype, Vtype=Vtype, alpha1=alpha_1, beta1=beta_1, alpha2=alpha_2, beta2=beta_2)
    y_test = torch.bmm(A_test, x_test.unsqueeze(2)).squeeze(2)
    for m in m_ls:
        Ym = generate_Ym_small(M, m, d, tmp_alpha_3, tmp_beta_3)
        Am_test = A_test @ Ym

        #tmp_data = [x_test, y_test, A_test, Am_test]
        tmp_data = [x_test.detach().cpu().numpy(), y_test.detach().cpu().numpy(), A_test.detach().cpu().numpy(),
                    Am_test.detach().cpu().numpy()]

        test_data_ls.append(tmp_data)

data_name = f'Test_data_var_f_M_{M}_alp1_{alpha_1}_beta1_{beta_1}_alp2_{alpha_2}_beta2_{beta_2}_alp3_{alpha_3}_beta3_{beta_3}.pkl'
with open(data_name, 'wb') as f:
    pickle.dump(test_data_ls, f)

