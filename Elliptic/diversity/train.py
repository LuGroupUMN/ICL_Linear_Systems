from config import *
from utils.train_utils import *
import pickle

np.random.seed(925)
torch.manual_seed(925)

device = torch.device("cuda" if torch.cuda.is_available() else
                      "mps" if torch.backends.mps.is_available() else
                      "cpu")

#wandb.login(key='a31c2a5402e98f77d1ca61b14796dfcdeaa2ea5c')
d = config["d"]
l1, l2 = config["l1"], config["l2"]
m0, n, M, N = config["m0"], config["n"], config["M"], config["N"]
mul_ls = config["mul_ls"]
N_max = N + M
para_ls = config["para_ls"]
m_ls = m0 * mul_ls
Ltype, Vtype = config["Ltype"], config["Vtype"]
### Load data
data_name = f'Train_data_N_{N_max}_n_{n}_l1_{num2str_decimal(l1)}_l2_{num2str_decimal(l2)}_Ltype_{Ltype}_Vtype_{Vtype}.pkl'
with open(data_name, 'rb') as f:
    data = pickle.load(f)

### training
x_train = data['x_train']
y_train = data['y_train']
A = data['A']
An = data['An']

A, An, x_train, y_train = (torch.tensor(arr, dtype=torch.float32, device=device) for arr in
                               (A, An, x_train, y_train))
# ### train op
# project_name = f'Div_op_N_{N}_n_{n}'
# wandb.init(project=project_name, name='train', reinit=True, config={
#         "N": N,
#         "learning_rate": train_config["learning_rate"],
#         "decay_rate": train_config["decay_rate"],
#         "decay_epoch": train_config["decay_epoch"],
#         "epochs": train_config["epochs"],
#     })
# mdl = train_mdl(d, A, An, x_train, y_train, init_weight= 'op', num_epochs=train_config["epochs"], lr=train_config["learning_rate"],
#                     decay_epoch=train_config["decay_epoch"], decay_rate=train_config["decay_rate"])
# # Save the model
# model_save_path = f"trained_op_model_N_{N}_n_{n}.pt"
# torch.save(mdl, model_save_path)

### train bad
# project_name = f'Div_bad_N_{N}_n_{n}'
# wandb.init(project=project_name, name='train', reinit=True, config={
#         "N": N,
#         "learning_rate": train_config["learning_rate"],
#         "decay_rate": train_config["decay_rate"],
#         "decay_epoch": train_config["decay_epoch"],
#         "epochs": train_config["epochs"],
#     })
mdl = train_mdl(d, A, An, x_train, y_train, loss_type='h1', use_wandb=False, init_weight= 'bad', num_epochs=train_config["epochs"], lr=train_config["learning_rate"],
                    decay_epoch=train_config["decay_epoch"], decay_rate=train_config["decay_rate"])
# Save the model
model_save_path = f"trained_bad_model_N_{N}_n_{n}.pt"
torch.save(mdl, model_save_path)
