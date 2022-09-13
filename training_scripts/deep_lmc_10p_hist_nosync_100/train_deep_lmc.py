from deep_lmc_natural_gradients import DeepLMC
from gpytorch.mlls import DeepApproximateMLL, VariationalELBO
import gpytorch, torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import os, time
import matplotlib.pyplot as plt

arg = int(os.environ['SLURM_PROCID'])
ntasks = int(os.environ['SLURM_NTASKS'])
time.sleep(arg)

batch_size = 200
n_train = 100

data_path = os.path.join('../../simulation_scripts')

psds = np.load(os.path.join(data_path, '10p_hist_psd_nosync.npy')).astype(np.float32)
labels = np.load(os.path.join(data_path, '10p_labels_nosync.npy')).astype(np.float32)
psds = np.concatenate([psds[:,0,:], psds[:,1,:]], axis=1)

minlabels = np.array([1.0, 4.5, 1., 15., 0.1, 0.0, 25., 15., 100., 0.])
difflabels = np.array([2.5, 3.5, 7., 15., 3.9, 5., 75., 10., 200., 10.])
labels -= minlabels
labels /= difflabels

print(labels.max(axis=0))
print(labels.min(axis=0))

X_train = labels[:n_train].copy()
Y_train = np.log10(psds[:n_train].copy())
Yn_train = Y_train - Y_train.mean(axis=0, keepdims=True)

X_test = labels[3000:].copy()
Y_test = np.log10(psds[3000:].copy())
Yn_test = Y_test - Y_train.mean(axis=0, keepdims=True)

train_x = torch.tensor(X_train)
train_y = torch.tensor(Yn_train)

test_x = torch.tensor(X_test)
test_y = torch.tensor(Yn_test)

train_dataset = TensorDataset(train_x, train_y)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = TensorDataset(test_x, test_y)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

num_tasks = train_y.size(-1)
hidden_dims= 10
num_latent_gps = 14
num_inducing1 = 128
num_inducing2 = 128

model = DeepLMC(train_x.shape,
                num_tasks=num_tasks,
                num_hidden_dgp_dims=hidden_dims,
                num_latent_gps=num_latent_gps,
                num_inducing1=num_inducing1,
                num_inducing2=num_inducing2)

mll = DeepApproximateMLL(VariationalELBO(model.likelihood, model, num_data=train_y.size(0)))
history = []

variational_lrs = [1e-4] * 3 + [1e-3] * 3 + [1e-2] * 3
hyperparam_lrs = [1e-4, 1e-3, 1e-2] * 3
variational_ngd_optimizer = gpytorch.optim.NGD(model.variational_parameters(), num_data=train_y.size(0), lr=variational_lrs[arg])
hyperparameter_optimizer = torch.optim.Adam([
    {'params': model.hyperparameters()}], lr=hyperparam_lrs[arg])
abort = False
savefile = f'save_{arg}.txt'
with open(savefile, 'a') as f:
    f.writelines(f'{variational_lrs[arg]},{hyperparam_lrs[arg]},{num_inducing1},{num_inducing2}\n')
num_epochs = 16000
min_loss = 1e10
history = []
for i in range(num_epochs):
    loss_avg = 0.
    batch_counter = 0.
    for x_batch, y_batch in train_loader:
        variational_ngd_optimizer.zero_grad()
        hyperparameter_optimizer.zero_grad()
        try:
            output = model(x_batch)
        except:
            abort = True
            break
        loss = -mll(output, y_batch)
        history.append(loss.item())
        loss_avg += loss.item()
        batch_counter += 1.
        loss.backward()
        variational_ngd_optimizer.step()
        hyperparameter_optimizer.step()
    loss_avg = loss_avg / batch_counter
    if abort:
        break
    if i > 200 and loss_avg < min_loss:
        torch.save(model.state_dict(), 'deep_lmc_%d.pth'%arg)
        min_loss = loss_avg
    if i % 100 ==  0:
        with torch.no_grad():
            test_errors = []
            for x_batch, y_batch in test_loader:
                output = model.likelihood(model(x_batch))
                test_mean = output.mean.mean(0)
                test_error = np.abs(test_mean - y_batch)
                test_errors.append(test_error.numpy())
            test_errors = np.concatenate(test_errors)

            train_errors = []
            for x_batch, y_batch in train_loader:
                output = model.likelihood(model(x_batch))
                train_mean = output.mean.mean(0)
                train_error = np.abs(train_mean - y_batch)
                train_errors.append(train_error.numpy())
            train_errors = np.concatenate(train_errors)

            print('###############################', flush=True)
            print('RANK %d step %d:'%(arg, i), flush=True)
            print('min_loss = %.4f'%min(history))
            print('train_error = %.4f'%train_errors.mean(), flush=True)
            print('test_error = %.4f'%test_errors.mean(), flush=True)
            print('###############################', flush=True)
            with open(savefile, 'a') as f:
                f.writelines(f'{history[0]},{train_errors.mean()},{test_errors.mean()}\n')
                f.writelines([f'{h},,\n' for h in history[1:]])
            history = []
