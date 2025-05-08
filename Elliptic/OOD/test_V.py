from utils.train_utils import *
from config import *
import pickle

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

### test vary a
data_name = f'Test_data_var_V_M_{M}_alp1_{alpha_1}_beta1_{beta_1}_alp2_{alpha_2}_beta2_{beta_2}_alp3_{alpha_3}_beta3_{beta_3}.pkl'
with open(data_name, 'rb') as f:
    test_data_ls = pickle.load(f)

# model_save_path = f"trained_model_N_{N}_n_{n}_alp1_{alpha_1}_beta1_{beta_1}_alp2_{alpha_2}_beta2_{beta_2}_alp3_{alpha_3}_beta3_{beta_3}.pt"
# mdl = torch.load(model_save_path, weights_only=False, map_location=device)
mdl = PQModelop_lognormal(d).to(device)
l2_ls, h1_ls = [], []
for tmp_data in test_data_ls:
    x_test, y_test, A_test, Am_test = tmp_data
    x_test, y_test, A_test, Am_test = (torch.tensor(arr, dtype=torch.float32, device=device) for arr in
                                       (x_test, y_test, A_test, Am_test))

    _, l2, h1 = eval_mdl(mdl, A_test, Am_test, x_test, y_test)
    l2_ls.append(l2)
    h1_ls.append(h1)

l2_vec = np.array(l2_ls)
l2_mat = l2_vec.reshape(len(para_ls), len(mul_ls))

h1_vec = np.array(h1_ls)
h1_mat = h1_vec.reshape(len(para_ls), len(mul_ls))

npy_name = 'Elliptic_Result_OOD_test_V.npy'
with open(npy_name, 'wb') as s:
    np.save(s, m_ls)
    np.save(s, para_ls)
    np.save(s, l2_mat)
    np.save(s, h1_mat)

# import matplotlib.pyplot as plt
# plt.figure(1)
# plt.loglog(m_ls, l2_mat[0, ...], label=rf'$(\alpha_2,\beta_2)=$(' + str(para_ls[0][0])+','+str(para_ls[0][1])+')')
# for i in range(1, len(para_ls)//2+1):
#     plt.loglog(m_ls, l2_mat[i, ...], '--', label=rf'$(\alpha_2,\beta_2)=$(' + str(para_ls[i][0])+','+str(para_ls[i][1])+')')
#
# plt.title(f'd={d},n={n},N={20000}', fontsize=18)
# plt.xlabel(r'$m$', fontsize=18)
# plt.ylabel(r'Relative $L^2$', fontsize=18)
# # plt.ylim([1e-1, 1e0])
# plt.xticks(fontsize=12)
# plt.yticks(fontsize=12)
# plt.legend(fontsize=12)
# plt.savefig(f'test_V_L2_1_alp1_{alpha_1}_beta1_{beta_1}_alp2_{alpha_2}_beta2_{beta_2}_alp3_{alpha_3}_beta3_{beta_3}.png', bbox_inches = 'tight')
#
# plt.figure(2)
# plt.loglog(m_ls, l2_mat[0, ...], label=rf'$(\alpha_2,\beta_2)=$(' + str(para_ls[0][0])+','+str(para_ls[0][1])+')')
# for i in range(len(para_ls)//2+1, len(para_ls)):
#     plt.loglog(m_ls, l2_mat[i, ...], '--', label=rf'$(\alpha_2,\beta_2)=$(' + str(para_ls[i][0])+','+str(para_ls[i][1])+')')
#
# plt.title(f'd={d},n={n},N={20000}', fontsize=18)
# plt.xlabel(r'$m$', fontsize=18)
# plt.ylabel(r'Relative $L^2$', fontsize=18)
# plt.xticks(fontsize=12)
# plt.yticks(fontsize=12)
# plt.legend(fontsize=12)
# plt.savefig(f'test_V_L2_2_alp1_{alpha_1}_beta1_{beta_1}_alp2_{alpha_2}_beta2_{beta_2}_alp3_{alpha_3}_beta3_{beta_3}.png', bbox_inches = 'tight')
#
#
# plt.figure(3)
# plt.loglog(m_ls, h1_mat[0, ...], label=rf'$(\alpha_2,\beta_2)=$(' + str(para_ls[0][0])+','+str(para_ls[0][1])+')')
# for i in range(1, len(para_ls)//2+1):
#     plt.loglog(m_ls, h1_mat[i, ...], '--', label=rf'$(\alpha_2,\beta_2)=$(' + str(para_ls[i][0])+','+str(para_ls[i][1])+')')
#
# plt.title(f'd={d},n={n},N={20000}', fontsize=18)
# plt.xlabel(r'$m$', fontsize=18)
# plt.ylabel(r'Relative $H1$', fontsize=18)
# # plt.ylim([1e-1, 1e0])
# plt.xticks(fontsize=12)
# plt.yticks(fontsize=12)
# plt.legend(fontsize=12)
# plt.savefig(f'test_V_h1_1_alp1_{alpha_1}_beta1_{beta_1}_alp2_{alpha_2}_beta2_{beta_2}_alp3_{alpha_3}_beta3_{beta_3}.png', bbox_inches = 'tight')
#
# plt.figure(4)
# plt.loglog(m_ls, l2_mat[0, ...], label=rf'$(\alpha_2,\beta_2)=$(' + str(para_ls[0][0])+','+str(para_ls[0][1])+')')
# for i in range(len(para_ls)//2+1, len(para_ls)):
#     plt.loglog(m_ls, l2_mat[i, ...], '--', label=rf'$(\alpha_2,\beta_2)=$(' + str(para_ls[i][0])+','+str(para_ls[i][1])+')')
#
# plt.title(f'd={d},n={n},N={20000}', fontsize=18)
# plt.xlabel(r'$m$', fontsize=18)
# plt.ylabel(r'Relative $H_1$', fontsize=18)
# plt.xticks(fontsize=12)
# plt.yticks(fontsize=12)
# plt.legend(fontsize=12)
# plt.savefig(f'test_V_H1_2_alp1_{alpha_1}_beta1_{beta_1}_alp2_{alpha_2}_beta2_{beta_2}_alp3_{alpha_3}_beta3_{beta_3}.png', bbox_inches = 'tight')
#
#
#
#
#
# plt.show()