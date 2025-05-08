from train_utils import *
from config import *
import wandb
import pickle

device = torch.device("cuda" if torch.cuda.is_available() else
                      "mps" if torch.backends.mps.is_available() else
                      "cpu")

wandb.login(key='a31c2a5402e98f77d1ca61b14796dfcdeaa2ea5c')
d = config["d"]
para_ls = config["para_ls"]
alpha_1, beta_1, alpha_2, beta_2, alpha_3, beta_3 = config["alpha_1"], config["beta_1"], config["alpha_2"], config["beta_2"], config["alpha_3"], config["beta_3"]
m0, n, M, N = config["m0"], config["n"], config["M"], config["N"]
mul_ls = config["mul_ls"]
num_quad = config["num_quad"]
N_max = N + M

m_ls = m0 * mul_ls


project_name = f'ICL_OOD_N_{N}_n_{n}_alp1_{alpha_1}_beta1_{beta_1}_alp2_{alpha_2}_beta2_{beta_2}_alp3_{alpha_3}_beta3_{beta_3}'
wandb.init(project=project_name, name='train', reinit=True, config={
        "N": N,
        "learning_rate": train_config["learning_rate"],
        "decay_rate": train_config["decay_rate"],
        "decay_epoch": train_config["decay_epoch"],
        "epochs": train_config["epochs"],
    })

### load data
data_name = f'Train_data_N_{N}_n_{n}_alp1_{alpha_1}_beta1_{beta_1}_alp2_{alpha_2}_beta2_{beta_2}_alp3_{alpha_3}_beta3_{beta_3}.pkl'
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
model_save_path = f"trained_model_N_{N}_n_{n}_alp1_{alpha_1}_beta1_{beta_1}_alp2_{alpha_2}_beta2_{beta_2}_alp3_{alpha_3}_beta3_{beta_3}.pt"
torch.save(mdl, model_save_path)






