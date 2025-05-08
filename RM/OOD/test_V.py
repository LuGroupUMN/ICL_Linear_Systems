from utils.train_utils import *
from config import *
import pickle

device = torch.device("cuda" if torch.cuda.is_available() else
                      "mps" if torch.backends.mps.is_available() else
                      "cpu")

d = config["d"]
para_f = config["para_f"]
para_V = config["para_V"]
m0, n, M, N = config["m0"], config["n"], config["M"], config["N"]
mul_ls = config["mul_ls"]
num_quad = config["num_quad"]
N_max = N + M
m_ls = m0 * mul_ls

### test vary a
data_name = f'Test_data_var_v_M_{M}.pkl'
with open(data_name, 'rb') as f:
    test_data_ls = pickle.load(f)

# model_save_path = f"trained_model_N_{N}_n_{n}.pt"
# mdl = torch.load(model_save_path, weights_only=False, map_location=device)

mdl = PQModel(d).to(device)


l2_ls, h1_ls = [], []
for tmp_data in test_data_ls:
    x_test, y_test, A_test, Am_test = tmp_data
    x_test, y_test, A_test, Am_test = (torch.tensor(arr, dtype=torch.float32, device=device) for arr in
                                       (x_test, y_test, A_test, Am_test))

    _, l2, h1 = eval_mdl(mdl, A_test, Am_test, x_test, y_test)
    l2_ls.append(l2)
    h1_ls.append(h1)

l2_vec = np.array(l2_ls)
l2_mat = l2_vec.reshape(len(para_V), len(mul_ls))

h1_vec = np.array(h1_ls)
h1_mat = h1_vec.reshape(len(para_V), len(mul_ls))

npy_name = 'RM_Result_OOD_test_V.npy'
with open(npy_name, 'wb') as s:
    np.save(s, m_ls)
    np.save(s, para_V)
    np.save(s, l2_mat)

# import matplotlib.pyplot as plt
# plt.figure(1)
# plt.loglog(m_ls, l2_mat[0, ...], label=rf'$(a,b)=$(' + str(para_V[0][0])+','+str(para_V[0][1])+')')
# for i in range(1, len(para_V)):
#     plt.loglog(m_ls, l2_mat[i, ...], '--', label=rf'$(a,b)=$(' + str(para_V[i][0])+','+str(para_V[i][1])+')')
#
# plt.title(f'd={d},n={n},N={20000}', fontsize=18)
# plt.xlabel(r'$m$', fontsize=18)
# plt.ylabel(r'Relative $L^2$', fontsize=18)
# plt.xticks(fontsize=12)
# plt.yticks(fontsize=12)
# plt.legend(fontsize=12)
# plt.savefig(f'test_V.png', bbox_inches = 'tight')
# plt.show()