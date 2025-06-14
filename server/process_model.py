import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import sys, os
from scipy.ndimage import gaussian_filter1d

if '/home/new_lab/test/ensemble_bae' not in sys.path:
    sys.path.append('/home/new_lab/test/ensemble_bae')
if '/home/new_lab/test/ensemble_bae/server' not in sys.path:
    sys.path.append('/home/new_lab/test/ensemble_bae/server')

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from model import VAEModule, AutoEncoder
from dataloader import VAEReconLoader, VAESegLoader
from dataprovider import load_SMAP_P1, load_NAB_Texi2, load_SWAT_PIT502
from bae_ensemble import BAE_Ensemble
import utils

from store import DataStore

data_store = DataStore()

# model config

class Config:
    # model
    seq_len = 10
    in_channel = 1
    win_size = 96
    latent_dim = 10
    hidden_num_units = 64
    sigma = 0.1
    sigma2_offset = 0.01
    hidden_num_units_lstm = 64
    num_layers_lstm = 1
    
    # training
    n_epochs = 15
    lr = 1e-3
    batch_size = 64
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

M = 30
config = Config()



def train(train_dataloader, valid_dataloader, bvae_ens, config):
    def valid(bvae_ens, data_loader):
        bvae_ens.eval()
        total_loss = []
        with torch.no_grad():
            for i, x in enumerate(data_loader):
                x = x.view(config.batch_size, config.in_channel, -1)
                x = x.to(config.device)
                loss = bvae_ens.fit(x).to(config.device)
                total_loss.append(loss.item())
        return np.average(total_loss)

    for epoch in range(config.n_epochs):
        bvae_ens.train()
        print(f"-------------- epoch: {epoch} ----------------")
        total_loss = []
        for i, x in enumerate(train_dataloader):
            bvae_ens.zero_optimizers()
            x = x.view(config.batch_size, config.in_channel, -1) # [batch_size, in_channel, win_size]
            x = x.to(config.device)
            loss = bvae_ens.fit(x)
            total_loss.append(loss.item())
            if (i + 1) % 50 == 0:
                print(f"     batch: {i + 1}, batch_loss: {loss.item()}     ")
            
            loss.backward()
            bvae_ens.step_optimizers()
        train_loss = np.average(total_loss)
        valid_loss = valid(bvae_ens, valid_dataloader)
        print(f">>>> Epoch: {epoch}, train_loss: {train_loss}, valid_loss: {valid_loss} <<<<")


def predict_alea(bvae_ens, cmd=False):
    test = data_store.get_data("test")

    test_recons_dataset = VAEReconLoader(test, config.win_size, "test")

    batch_size = 1
    test_recons_dataloader =  DataLoader(
        dataset = test_recons_dataset, 
        batch_size = batch_size,
        shuffle=False,
        drop_last=True,
    )


    approx_var_heter = []
    bvae_ens.eval()

    for i, x in enumerate(test_recons_dataloader):
        x = x.view(batch_size, config.in_channel, -1)
        x = x.to(config.device)
        prediction, logvar = bvae_ens.predict_mean_logvar(x)
        var = logvar
        approx_var_heter.extend(var.detach().cpu().numpy().flatten())


    approx_var_heter = np.array(approx_var_heter)
    approx_var_heter1 = utils.remove_outliers_iqr(approx_var_heter, factor=2)
    if cmd:
        return approx_var_heter
    else:
        alea_smoot = gaussian_filter1d(approx_var_heter1, sigma=10)
        result = [{"x": i, "y": approx_var_heter1[i]} for i in range(len(approx_var_heter1))]
        # ds.add_data("alea", approx_var_heter)
        return result

# def predict_mult_epis(bvae_ens, cmd=False, count_epis=[10, 30, 60], index=None):
#     test = data_store.get_data("test")
#     test_recons_dataset = VAEReconLoader(test, config.win_size, "test")
#     batch_size = 1
#     test_dataloader =  DataLoader(
#         dataset = test_recons_dataset, 
#         batch_size = 1,
#         shuffle=False,
#         drop_last=True,
#     )
#     data_len = len(test_dataloader) * config.win_size
#     mult_epis = []


#     for ensemble_count in count_epis:
#         var_of_win = np.zeros((data_len, ensemble_count))
#         rec_signals = np.zeros((data_len, ensemble_count))

#         bvae_ens.eval()
#         for i, x in enumerate(test_recons_dataloader):
#             x = x.view(batch_size, config.in_channel, -1)
#             x = x.to(config.device)
#             predictions = bvae_ens.predict_stack_values(x) # (1, batch_size, in_channel, win_size)
#             for m in range(ensemble_count):
#                 rec_signals[i*config.win_size: (i+1)*config.win_size, m] = predictions[m].flatten().detach().cpu().numpy()

#         rec_signals_std = np.std(rec_signals, axis=1)
#         epis: np.ndarray = rec_signals_std
#         epis_smoot = gaussian_filter1d(epis, sigma=15)
#         mult_epis.append(epis_smoot)
#     mult_epis = np.array(mult_epis)

#     if cmd:
#         return epis
#     else:
#         result = {}
#         for i in range(len(count_epis)):
#             if index == None:
#                 result[f"epis_{i+1}"] = [{"x": j, "y": mult_epis[i][j]} for j in range(len(mult_epis[i]))]
#             else:
#                 result[f"epis_{i+1}"] = [{"x": j, "y": mult_epis[i][j]} for j in range(index[0], index[1])]
#         # ds.add_data("epis", epis)
#         return result
    
def predict_mult_epis(bvae_ens, cmd=False, count_epis=[5, 10, 30], index=None):
    test = data_store.get_data("test")
    test_recons_dataset = VAEReconLoader(test, config.win_size, "test")
    batch_size = 1
    test_recons_dataloader =  DataLoader(
        dataset = test_recons_dataset, 
        batch_size = 1,
        shuffle=False,
        drop_last=True,
    )
    data_len = len(test_recons_dataloader) * config.win_size
    mult_epis = []

    result = {
        "epis_upper_50": [[], [], []],
        "epis_lower_50": [[], [], []],
        "epis_upper_95": [[], [], []],
        "epis_lower_95": [[], [], []],
        "epis": [[], [], []]
    }
    for k, ensemble_count in enumerate(count_epis):
        var_of_win = np.zeros((data_len, ensemble_count))
        rec_signals = np.zeros((data_len, ensemble_count))

        bvae_ens.eval()
        for i, x in enumerate(test_recons_dataloader):
            x = x.view(batch_size, config.in_channel, -1)
            x = x.to(config.device)
            predictions = bvae_ens.predict_stack_values(x) # (1, batch_size, in_channel, win_size)
            for m in range(ensemble_count):
                rec_signals[i*config.win_size: (i+1)*config.win_size, m] = predictions[m].flatten().detach().cpu().numpy()

        
        rec_signals_mean = np.mean(rec_signals, axis=1)
        epis = np.std(rec_signals, axis=1)
        epis_smoot = gaussian_filter1d(epis, sigma=15)
        epis_upper_50 = np.convolve(rec_signals_mean + 0.67 * epis, np.ones(10)/10, mode='same')
        epis_lower_50 = np.convolve(rec_signals_mean - 0.67 * epis, np.ones(10)/10, mode='same')
        epis_upper_95 = np.convolve(rec_signals_mean + 1.96 * epis, np.ones(10)/10, mode='same')
        epis_lower_95 = np.convolve(rec_signals_mean - 1.96 * epis, np.ones(10)/10, mode='same')


   
        result["epis_upper_50"][k] = [{"x": j, "y": epis_upper_50[j]} for j in range(index[0], index[1])]
        result["epis_lower_50"][k] = [{"x": j, "y": epis_lower_50[j]} for j in range(index[0], index[1])]
        result["epis_upper_95"][k] = [{"x": j, "y": epis_upper_95[j]} for j in range(index[0], index[1])]
        result["epis_lower_95"][k] = [{"x": j, "y": epis_lower_95[j]} for j in range(index[0], index[1])]
        result["epis"][k] = [{"x": j, "y": epis_smoot[j]} for j in range(index[0], index[1])] # [{"x": i, "y": epis_smoot[i]} for i in range(len(epis))]

        # ds.add_data("epis", epis)
    return result


def predict_epis(bvae_ens, cmd=False):
    test = data_store.get_data("test")
    test_recons_dataset = VAEReconLoader(test, config.win_size, "test")
    batch_size = 1
    test_dataloader =  DataLoader(
        dataset = test_recons_dataset, 
        batch_size = 1,
        shuffle=False,
        drop_last=True,
    )


    data_len = len(test_dataloader) * config.win_size
    var_of_win = np.zeros((data_len, M))
    rec_signals = np.zeros((data_len, M))

    bvae_ens.eval()
    for i, x in enumerate(test_dataloader):
        x = x.view(batch_size, config.in_channel, -1)
        x = x.to(config.device)
        predictions = bvae_ens.predict_stack_values(x) # (1, batch_size, in_channel, win_size)

        for m in range(M):
            rec_signals[i*config.win_size: (i+1)*config.win_size, m] = predictions[m].flatten().detach().cpu().numpy()

    
    rec_signals_mean = np.mean(rec_signals, axis=1)
    rec_signals_std = np.std(rec_signals, axis=1)
    epis_upper_50 = rec_signals_mean + 0.67 * rec_signals_std
    epis_lower_50 = rec_signals_mean - 0.67 * rec_signals_std
    epis_upper_95 = rec_signals_mean + 1.96 * rec_signals_std
    epis_lower_95 = rec_signals_mean - 1.96 * rec_signals_std
    epis_upper_50 = np.convolve(epis_upper_50, np.ones(10)/10, mode='same')
    epis_lower_50 = np.convolve(epis_lower_50, np.ones(10)/10, mode='same')
    epis_upper_95 = np.convolve(epis_upper_95, np.ones(10)/10, mode='same')
    epis_lower_95 = np.convolve(epis_lower_95, np.ones(10)/10, mode='same')

    epis: np.ndarray = rec_signals_std

    if cmd:
        return epis
    else:
        epis_smoot = gaussian_filter1d(epis, sigma=10)
        result = {}
        result["epis_upper_50"] = [{"x": i, "y": epis_upper_50[i]} for i in range(len(epis_upper_50))]
        result["epis_lower_50"] = [{"x": i, "y": epis_lower_50[i]} for i in range(len(epis_lower_50))]
        result["epis_upper_95"] = [{"x": i, "y": epis_upper_95[i]} for i in range(len(epis_upper_95))]
        result["epis_lower_95"] = [{"x": i, "y": epis_lower_95[i]} for i in range(len(epis_upper_95))]
        result["epis"] = [{"x": i, "y": epis_smoot[i]} for i in range(len(epis))]
        # ds.add_data("epis", epis)
        return result



def predict_error(bvae_ens, cmd=False):
    # anomaly detection only vae on test
    test = data_store.get_data("test")
    test_dataset = VAESegLoader(test, config.win_size, "test")

    test_dataloader =  DataLoader(
        dataset = test_dataset, 
        batch_size = config.batch_size,
        shuffle=False,
        drop_last=True,
    )

    
    # for visualization
    data_len = len(test_dataloader) * config.batch_size
    error = np.zeros((data_len, M))

    bvae_ens.eval()
    for i, x in enumerate(test_dataloader):
        x = x.view(config.batch_size, config.in_channel, -1)
        x = x.to(config.device)
        predictions = bvae_ens.predict_stack_values(x) # (10, batch_size, in_channel, win_size)
        
        for m in range(M):
            error[i*config.batch_size: (i+1)*config.batch_size, m] = torch.mean((x - predictions[m])**2, dim=(1,2)).detach().cpu().numpy()


    error_mean = np.mean(error, axis=1)
    filter_win = 10
    # error_smoothed = np.convolve(error_mean, np.ones(filter_win)/filter_win, mode='same')
    error_smoothed = gaussian_filter1d(error_mean, sigma=15)
    prefix = np.ones(config.win_size - 1) * error_smoothed[0]
    error_smoothed = np.concatenate((prefix, error_smoothed))
    # upper_bound = np.mean(error, axis=1) + 2 * error_std
    # lower_bound = np.mean(error, axis=1) - 2 * error_std

    threshold = anomaly_threshold(bvae_ens)
    predict_label = np.zeros(error_smoothed.shape)
    predict_label[error_smoothed > threshold] = 1
    anomaly_index = utils.get_anomaly_segment(predict_label)
    anoamy_score = []
    for index in anomaly_index:
        # anoamy_score.append(np.sum((error_smoothed[index[0]: index[1]] - threshold)))
        anoamy_score.append(np.mean(error_mean[index[0]: index[1]]))
    anoamly_score = np.array(anoamy_score)
    if len(anoamly_score) > 1:
        anoamly_score = (anoamly_score - np.min(anoamly_score)) / (np.max(anoamly_score) - np.min(anoamly_score))
    
    if cmd == True:
        return error_smoothed
    else:
        result = [{"x1": index[0], "x2": index[1], "anomaly_score": anoamly_score[i]} for (i, index) in enumerate(anomaly_index)]
        return result

def load_Dataset(dataset: str):
    if dataset == "SMAP_P1":
        training, test, valid = load_SMAP_P1(abspath=True)
    elif dataset == "NY_Taxi":
        training, test, valid = load_NAB_Texi2(abspath=True)
    elif dataset == "SWAT_PIT502":
        training, test, valid = load_SWAT_PIT502(abspath=True)

    data_store.add_data("training", training)
    data_store.add_data("test", test)
    data_store.add_data("valid", valid)

def run(dataset: str, save=True):
    if dataset == "SMAP_P1":
        training, test, valid = load_SMAP_P1(abspath=True)
    elif dataset == "NY_Taxi":
        training, test, valid = load_NAB_Texi2(abspath=True)
    elif dataset == "SWAT_PIT502":
        training, test, valid = load_SWAT_PIT502(abspath=True)
    else:
        print(Exception(f"Dataset {dataset} not supported"))
        return
    
    train_dataset = VAESegLoader(training, config.win_size, "train")
    valid_dataset = VAESegLoader(valid, config.win_size, "valid")

    train_dataloader =  DataLoader(
        dataset = train_dataset, 
        batch_size = config.batch_size,
        shuffle=True,
        drop_last=True,
    )

    valid_dataloader =  DataLoader(
        dataset = valid_dataset, 
        batch_size = config.batch_size,
        shuffle=False,
        drop_last=True,
    )

    # training model
    bvae_ens = BAE_Ensemble(M, AutoEncoder, config)
    bvae_ens.toDevice(config.device)

    # training
    config.n_epochs = 5
    train(train_dataloader, valid_dataloader, bvae_ens, config)
    bvae_ens.set_loss_type("rnll")
    print(">>>> tunning var <<<<")
    bvae_ens.tunning_var(1e-3)
    config.n_epochs = 5
    train(train_dataloader, valid_dataloader, bvae_ens, config)
    if save:
        bvae_ens.save()
    return bvae_ens

def predict(bvae_ens, dataset: str = "SMAP_P1") -> DataStore:
    if dataset == "SMAP_P1":
        training, test, valid = load_SMAP_P1(abspath=True)
    # get aleatoric
    anoamly_score = predict_error(bvae_ens, test)
    epis = predict_epis(bvae_ens, test)
    alea = predict_alea(bvae_ens, test)

    return anoamly_score, epis, alea

def init_model():
    bvae_ensemble = BAE_Ensemble(M, AutoEncoder, config)
    bvae_ensemble.load()
    bvae_ensemble.toDevice(config.device)
    bvae_ensemble.eval()
    return bvae_ensemble

def anomaly_threshold(bvae_ens):
    training = data_store.get_data("training")
    train_dataset = VAESegLoader(training, config.win_size, "train")
    train_dataloader =  DataLoader(
        dataset = train_dataset, 
        batch_size = config.batch_size,
        shuffle=True,
        drop_last=True,
    )

    data_len = len(train_dataloader) * config.batch_size
    train_nll = np.zeros((data_len, M))

    # get train nlll -> error score threshold
    bvae_ens.eval()
    for i, x in enumerate(train_dataloader):
        x = x.view(config.batch_size, config.in_channel, -1)
        x = x.to(config.device)
        predictions = bvae_ens.predict_stack_values(x) # (10, batch_size, in_channel, win_size)
        for m in range(M):
            train_nll[i*config.batch_size: (i+1)*config.batch_size, m] = torch.mean((x - predictions[m])**2, dim=(1,2)).detach().cpu().numpy()

    train_nll_mean = np.mean(train_nll, axis=1)
    threshold = np.percentile(train_nll_mean, 96) # smap 90
    return threshold

if __name__ == "__main__":
    load_Dataset("SMAP_P1")
    # load_Dataset("SWAT")
    bvae_ens = run("SMAP_P1", save=True)

    # bvae_ens = init_model()
    # training, test, valid = load_SMAP_P1(abspath=True)

    # test = data_store.get_data("test")
    # anoamly_score = predict_error(bvae_ens, cmd=True)
    # epis = predict_epis(bvae_ens, cmd=True)
    # alea = predict_alea(bvae_ens, cmd=True)


    # # threshold = np.percentile(anoamly_score, 85)
    # threshold = anomaly_threshold(bvae_ens)

    # import matplotlib.pyplot as plt

    # fig, ax = plt.subplots(nrows=4, ncols=1, figsize=(12, 12))
    # ax[0].plot(test.flatten()[: len(anoamly_score)], label='original signal')
    # # for index in anomaly_index:
    # #     ax[0].fill_between(np.arange(index[0], index[1]), -1, 3, color='red', alpha=0.2)
    # alea_smoot = gaussian_filter1d(alea, sigma=15)

    # ax[1].plot(alea_smoot, label = 'alea uncertaity')
    # ax[1].axhline(y = np.percentile(alea, 90), color='r', linestyle='--', label='alea threshold')
    # # ax[1].set_ylim(0, 0.4)
    # ax[1].legend()
    # ax[2].plot(epis, label = 'epic uncertainty')
    # # ax[2].plot(epic_train_smoot, label = "train epic")
    # # ax[2].set_ylim(0, 0.4)
    # ax[2].legend()
    # ax[3].plot(anoamly_score, label='anomaly score')
    # ax[3].axhline(y=threshold, color='r', linestyle='--', label='threshold')
    # ax[3].legend()
    # plt.show()
