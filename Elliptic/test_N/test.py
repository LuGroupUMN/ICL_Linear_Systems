from utils.train_utils import *
from config import *
import wandb
import pickle

np.random.seed(925)
torch.manual_seed(925)

device = torch.device("cuda" if torch.cuda.is_available() else
                      "mps" if torch.backends.mps.is_available() else
                      "cpu")

d, l1, l2 = config["d"], config["l1"], config["l2"]
n, M, N0 = config["n"], config["M"], config["N0"]
mul_ls = config["mul_ls"]
m_ls = config["m_ls"]
N_ls = N0 * mul_ls
N_ls = N_ls.astype(int)
N_max = N_ls[-1]
Ltype, Vtype = config["Ltype"], config["Vtype"]

data_name = f'Test_data_M_{M}_l1_{num2str_decimal(l1)}_l2_{num2str_decimal(l2)}_Ltype_{Ltype}_Vtype_{Vtype}.pkl'
with open(data_name, 'rb') as f:
    test_data_ls = pickle.load(f)

mse_ls = []
h1_ls = []
for N in N_ls:
    model_save_path = f"trained_model_test_N_var_m_N_{N}_n_{n}_l1_{num2str_decimal(l1)}_l2_{num2str_decimal(l2)}_Ltype_{Ltype}_Vtype_{Vtype}.pt"
    mdl = torch.load(model_save_path, weights_only=False, map_location=device)

    for tmp_data in test_data_ls:
        x_test, y_test, A_test, Am_test = tmp_data
        x_test, y_test, A_test, Am_test = (torch.tensor(arr, dtype=torch.float32, device=device) for arr in
                                           (x_test, y_test, A_test, Am_test))

        mse, l2e, h1 = eval_mdl(mdl, A_test, Am_test, x_test, y_test)

        mse_ls.append(mse)
        h1_ls.append(h1)

mse_vec = np.array(mse_ls)
mse_mat = mse_vec.reshape(len(N_ls), len(m_ls)).T

h1_vec = np.array(h1_ls)
h1_mat = h1_vec.reshape(len(N_ls), len(m_ls)).T

id = -3

ref_N = N_ls[0]
ref_y = mse_mat[0, 0] - mse_mat[0, -1]
scale_1 = ref_y / (ref_N ** (-1.0))
mse_ref_rate = 1e-1 * scale_1 * N_ls[:id]**(-1.0)


ref_N = N_ls[0]
ref_y = h1_mat[0, 0] - h1_mat[0, -1]
scale_1 = ref_y / (ref_N ** (-1.0))
h1_ref_rate = 1e-1 * scale_1 * N_ls[:id]**(-1.0)

N_ls = N_ls[:id]

mse_mat = mse_mat[:, :id] - mse_mat[:, [-1]]
h1_mat = h1_mat[:, :id] - h1_mat[:, [-1]]

npy_name = 'Elliptic_Result_test_N.npy'
with open(npy_name, 'wb') as s:
    np.save(s, N_ls)
    np.save(s, m_ls)
    np.save(s, mse_mat)
    np.save(s, mse_ref_rate)
    np.save(s, h1_mat)
    np.save(s, h1_ref_rate)



# print(mse_mat, h1_mat)
# zxc
#
# #mse_mat = mse_vec.reshape(len(m_ls), len(N_ls))
# import matplotlib.pyplot as plt
# plt.figure(1)
# for i in range(len(m_ls)):
#     plt.loglog(N_ls[:-3], mse_mat[i, :-3] - mse_mat[i, -1], '--*', label=f'm={m_ls[i]}')
#
# ref_N = N_ls[0]
# ref_y = mse_mat[0, 0] - mse_mat[0, -1]
# scale_2 = ref_y / (ref_N ** (-2.0))
# scale_1 = ref_y / (ref_N ** (-1.0))
# scale_h = ref_y / (ref_N ** (-0.5))
#
# #plt.loglog(N_ls[:-1], scale_2 * N_ls[:-1]**(-0.5*4), 'r', label=r'$O(N^{-2})$')
# plt.loglog(N_ls[:-3], 1e-1 * scale_1 * N_ls[:-3]**(-1.0), 'b', label=r'$O(N^{-1})$')
# #plt.loglog(N_ls[:-1], scale_h * N_ls[:-1]**(-0.5), 'c', label=r'$O(N^{-1/2})$')
#
#
# plt.title(f'd={d},n={n}', fontsize=18)
# plt.xlabel(r'$N$', fontsize=18)
# plt.ylabel(r'Shifted MSE', fontsize=18)
# plt.xticks(fontsize=12)
# plt.yticks(fontsize=12)
# plt.legend(fontsize=12)
# plt.savefig(f'test_N_mse_l1_{num2str_decimal(l1)}_l2_{num2str_decimal(l2)}_Ltype_{Ltype}_Vtype_{Vtype}.png', bbox_inches = 'tight')
#
#
# plt.figure(2)
# for i in range(len(m_ls)):
#     plt.loglog(N_ls[:-3], h1_mat[i, :-3] - h1_mat[i, -1], '--*', label=f'm={m_ls[i]}')
#
# ref_N = N_ls[0]
# ref_y = h1_mat[0, 0] - h1_mat[0, -1]
# scale_2 = ref_y / (ref_N ** (-2.0))
# scale_1 = ref_y / (ref_N ** (-1.0))
# scale_h = ref_y / (ref_N ** (-0.5))
#
# plt.loglog(N_ls[:-3], 1e-1 * scale_1 * N_ls[:-3]**(-0.5*2), 'b', label=r'$O(N^{-1})$')
#
# print(N_ls, scale_1)
# plt.title(f'd={d},n={n}', fontsize=18)
# plt.xlabel(r'$N$', fontsize=18)
# plt.ylabel(r'Shifted MSE', fontsize=18)
# plt.xticks(fontsize=12)
# plt.yticks(fontsize=12)
# plt.legend(fontsize=12)
# plt.savefig(f'test_N_h1_l1_{num2str_decimal(l1)}_l2_{num2str_decimal(l2)}_Ltype_{Ltype}_Vtype_{Vtype}.png', bbox_inches = 'tight')
# plt.show()
#

