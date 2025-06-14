import torch
import argparse
import numpy as np
import pandas as pd
import os
import random
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from model import VAEModule, AutoEncoder
from dataloader import VAESegLoader, VAEReconLoader
from bae_ensemble import BAE_Ensemble

import utils
import dataprovider as dp
import matplotlib.pyplot as plt


def get_args():
    parser = argparse.ArgumentParser(description=None)

  
    parser.add_argument('--n_epochs', default = 20, type = int)
    parser.add_argument('--ensemble_examples', default = 50, type = int)
    parser.add_argument('--batch_size', default = 32, type = int)
    parser.add_argument('--win_size', default=64, type=int)
    parser.add_argument('--learning_rate', default = 1e-3, type = float)

    parser.add_argument('--seed', default=1, type=int, help='random seed')
    parser.add_argument('--metric', default=False, type=bool, help='calculate mse between predicted and true aleatoric')
    parser.add_argument('--alea', type=float, help='aleatoric noise')
    parser.add_argument('--anomaly_value', type=float, help='anomaly value')
    parser.add_argument('--plot', default=False, type=bool, help='plot result')

    parser.add_argument('--lambda1', default=2, type=int, help="rnll hyperarg")
    parser.add_argument('--lambda2', default=1, type=int, help="rnll hyperarg")

    return parser.parse_args()

class Config:
    # model

    in_channel = 1
    win_size = 64
    latent_dim = 10
    hidden_num_units = 64
    sigma = 0.1
    sigma2_offset = 0.01
    # training
    n_epochs = 15
    lr = 1e-3
    batch_size = 16 # 32

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def predict_alea(test, bvae_ens):
    test_recons_dataset = VAEReconLoader(test, config.win_size, "test")

    batch_size = 1
    test_recons_dataloader =  DataLoader(
        dataset = test_recons_dataset, 
        batch_size = batch_size,
        shuffle=False,
        drop_last=True,
    )

    # for visualization
    recstructed_signal = []
    approx_var_heter = []

    bvae_ens.eval()

    for i, x in enumerate(test_recons_dataloader):
        x = x.view(batch_size, config.in_channel, -1)
        # prediction = bvae_ens.predict_mean(x) # (10, batch_size, in_channel, win_size)
        x = x.to(config.device)
        prediction, logvar = bvae_ens.predict_mean_logvar(x)
        var = logvar
        recstructed_signal.extend(prediction.detach().cpu().numpy().flatten())
        approx_var_heter.extend(var.detach().cpu().numpy().flatten())

    recstructed_signal = np.array(recstructed_signal)
    approx_var_heter = np.array(approx_var_heter)

    return approx_var_heter


def save_result(metrics, seed, alea, anomaly_value):
    '''
        metrics: List[float]. \\
        Includes "nll", "rnll", "beta_nll", "mts_nll"
    '''

    df = pd.DataFrame([[metrics[0], metrics[1], metrics[2], metrics[3], seed]], columns=['nll', 'rnll', 'beta_nll', 'mts_nll', 'seed'])
    filename = f'./toy_example_result_5_18/alea_{alea}_anomaly_{anomaly_value}_lambda1_{args.lambda1}_lambda2_{args.lambda2}.csv'
    mode = 'a' if  os.path.exists(filename) else 'w'
    header = mode == 'w'
    df.to_csv(filename, mode=mode, header=header, index=False)


def save_fig(approx_var_heter, true_var_heter, loss_type, anomaly_value):

    fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(4, 2))
    axs.set_ylim(0, 0.5)

    axs.plot(approx_var_heter, label='Predicted Aleatoric')
    axs.plot(true_var_heter, label='True Aleatoric')

    axs.legend()
    axs.set_xlabel("Time", fontsize=9)
    axs.set_ylabel("Aleatoric", fontsize=10)
    plt.savefig(f"toy_example_result_5_18/toy_example_{anomaly_value}_{loss_type}.png", bbox_inches='tight', dpi=300)

def preprocess():
    generator = utils.Synthetic_heteroscedasticData_generator(args.alea, 2, 5, args.anomaly_value)
    data = generator.get_data(1000)

    y_tildes = data['y_tilde'].clone().detach()
    y_tildes_test = data['y_disturb_tilde'].clone().detach()

    scaler = StandardScaler()
    scaler.fit(y_tildes)
    y_tildes = scaler.transform(y_tildes)
    y_tildes_test = scaler.transform(y_tildes_test)

    true_var_heter = ((generator.get_y_std(data['y_disturb']) / scaler.scale_)**2).detach().numpy().flatten()
    test = y_tildes_test
    training = y_tildes

    train_syn_dataset = VAESegLoader(training, config.win_size, "train")
    train_syn_dataloader =  DataLoader(
        dataset = train_syn_dataset, 
        batch_size = config.batch_size,
        shuffle=True,
        drop_last=True,
    )

    valid_syn_dataset = VAESegLoader(test, config.win_size, "test")
    valid_syn_dataloader =  DataLoader(
        dataset = valid_syn_dataset, 
        batch_size = config.batch_size,
        shuffle=False,
        drop_last=True,
    )

    return train_syn_dataloader, valid_syn_dataloader, true_var_heter, test

def run(args, loss_type, train_syn_dataloader, valid_syn_dataloader, true_var_heter, test):
   
    def valid(data_loader):
            bvae_ens.eval()
            total_loss = []
            with torch.no_grad():
                for i, x in enumerate(data_loader):
                    x = x.view(config.batch_size, config.in_channel, -1)
                    x = x.to(config.device)
                    loss = bvae_ens.fit(x)
                    total_loss.append(loss.item())
            return np.average(total_loss)


    def train(data_loader, config):
        bvae_ens.train()
        for epoch in range(config.n_epochs):
            print(f"-------------- epoch: {epoch} ----------------")
            total_loss = []
            for i, x in enumerate(data_loader):
                bvae_ens.zero_optimizers()
                x = x.view(config.batch_size, config.in_channel, -1) 
                x = x.to(config.device)
                loss = bvae_ens.fit(x)
                total_loss.append(loss.item())
                if (i + 1) % 20 == 0:
                    print(f"     batch: {i + 1}, batch_loss: {loss.item()}     ")
                loss.backward()
                bvae_ens.step_optimizers()
            train_loss = np.average(total_loss)
            valid_loss = valid(valid_syn_dataloader)
            print(f">>>> Epoch: {epoch}, train_loss: {train_loss}, valid_loss: {valid_loss} <<<<")



    bvae_ens = BAE_Ensemble(M, AutoEncoder, config)
    bvae_ens.toDevice(config.device)

    bvae_ens.set_loss_type(loss_type)
    if loss_type == 'rnll':
        bvae_ens.set_rnll_hyper(args.lambda1, args.lambda2)

    config.n_epochs = 5
    train(train_syn_dataloader, config)

    print(f" ******** training with var tunning ********")
    
    config.n_epochs = 10
    bvae_ens.tunning_var(args.learning_rate)
    train(train_syn_dataloader, config)

    # predict 
    approx_var_heter = predict_alea(test, bvae_ens)
   
    
    if args.plot:
        save_fig(approx_var_heter, true_var_heter, loss_type, args.anomaly_value)

   
    return np.mean((approx_var_heter - true_var_heter[:len(approx_var_heter)])**2) # mse between predicted and true aleatoric
        

if __name__ == "__main__":
    args = get_args()

    config = Config()
    M = 20
    metrics = []
    train_syn_dataloader, valid_syn_dataloader, true_var_heter, test = preprocess()

    for loss_type in ["nll", "rnll" , "beta_nll", "mts_nll"]:
        print(f" ******** loss_type: {loss_type} ******** ")
        mse = run(args, loss_type, train_syn_dataloader, valid_syn_dataloader, true_var_heter, test)
        metrics.append(mse)
    
    if args.metric:
        save_result(metrics, args.seed, args.alea, args.anomaly_value)
