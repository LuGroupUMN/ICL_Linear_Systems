from utils.train_utils import *
from config import *
import wandb
import pickle

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

project_name = f'RM_OOD_N_{N}_n_{n}'
wandb.init(project=project_name, name='train', reinit=True, config={
        "N": N,
        "learning_rate": train_config["learning_rate"],
        "decay_rate": train_config["decay_rate"],
        "decay_epoch": train_config["decay_epoch"],
        "epochs": train_config["epochs"],
    })

### load data
data_name = f'Train_data_N_{N}_n_{n}.pkl'
with open(data_name, 'rb') as f:
    data = pickle.load(f)

x_train = data['x_train']
y_train = data['y_train']
A = data['A']
An = data['An']

A, An, x_train, y_train = (torch.tensor(arr, dtype=torch.float32, device=device) for arr in
                               (A, An, x_train, y_train))

mdl = train_mdl(d, A, An, x_train, y_train, num_epochs=train_config["epochs"], lr=train_config["learning_rate"],
                    decay_epoch=train_config["decay_epoch"], decay_rate=train_config["decay_rate"])
# Save the model
model_save_path = f"trained_model_N_{N}_n_{n}.pt"
torch.save(mdl, model_save_path)






