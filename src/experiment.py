import torch
import argparse
import numpy as np
import pandas as pd
import os

from torch.utils.data import Dataset, DataLoader
from model import AutoEncoder
from dataloader import AESegLoader
from bae_ensemble import BAE_Ensemble

import utils
import dataprovider as dp
from datetime import datetime
import matplotlib.pyplot as plt
def get_args():
    parser = argparse.ArgumentParser(description=None)

  
    parser.add_argument('--n_epochs', default = 20, type = int)
    parser.add_argument('--ensemble_examples', default = 20, type = int)
    parser.add_argument('--batch_size', default = 64, type = int)
    parser.add_argument('--win_size', default=128, type=int)
    parser.add_argument('--learning_rate', default = 1e-3, type = float, help='learning rate for var tunning phase')
    parser.add_argument('--epochs2', default = 10, type=int, help='epochs for var tunning phase')
    parser.add_argument('--seed', default= 2, type=int, help='random seed')

    parser.add_argument('--dataset', type=str, help='dataset name')
    parser.add_argument('--loss_type', type=str, help='loss function')

    parser.add_argument('--lambda1', default=2, type=int, help="rnll hyperarg")
    parser.add_argument('--lambda2', default=1, type=int, help="rnll hyperarg")

    return parser.parse_args()


def get_anomaly_index(dataset: str):

    if dataset == "SMAP_P1":
        anomaly_index = [[150, 350], [1540, 1780], [2535, 2845]]
    elif dataset == "SMAP_E13":
        anomaly_index = [(300, 400), (600, 650), (1450, 1550)]
    elif dataset == "SMAP_E1":
        anomaly_index = [(500, 535), (1100, 1500)]
    elif dataset == "NAB_Ambient":
        anomaly_index = [[521, 522], [2980, 2981]]
    elif dataset == "NAB_taxi":
        anomaly_index = [[442, 443], [1683, 1684]]
    elif dataset == "NAB_Machine":
       anomaly_index = [[409, 410], [1986, 1987]]
    elif dataset == "SMD_machine2-1":
        anomaly_index = [[506, 530], [1900, 1960], [3340, 3380]] 
    elif dataset == "SMD_machine1-3":
        anomaly_index = [[372, 375], [1113, 1161]]
    elif dataset == "SMD_machine3-4":
        anomaly_index = [[234, 1020], [1974, 2050]]
    elif dataset == "MSL_P11":
        anomaly_index = [[1778, 1898], [1238, 1344]] 
    elif dataset == "MSL_F7":
        anomaly_index = [[1250, 1450], [2670, 2790], [3325, 3425]]
    elif dataset == "UCR_InternalBleeding16":
        anomaly_index = [[185, 200]]
    elif dataset == "UCR_InternalBleeding17":
        anomaly_index = [[2200,2310]]
    return anomaly_index


def get_dataloader(dataset: str):

    if dataset == "SMAP_P1":
        training, test, valid = dp.load_SMAP_P1()
    elif dataset == "SMAP_E13":
        training, test, valid = dp.load_SMAP_E13()
    elif dataset == "SMAP_E1":
        training, test, valid = dp.load_SMAP_E1()
    elif dataset == "NAB_Ambient":
        training, test, valid = dp.load_NAB_Ambient()
    elif dataset == "NAB_taxi":
        training, test, valid = dp.load_NAB_taxi()
    elif dataset == "NAB_Machine":
        training, test, valid = dp.load_NAB_Machine()
    elif dataset == "SMD_machine2-1":
        training, test, valid = dp.load_SMD_Machine2_1()
    elif dataset == "SMD_machine1-3":
        training, test, valid = dp.load_SMD_Machine1_3()
    elif dataset == "SMD_machine3-4":
        training, test, valid = dp.load_SMD_Machine3_4()
    elif dataset == "MSL_P11":
        training, test, valid = dp.load_MSL_P11()
    elif dataset == "MSL_F7":
        training, test, valid = dp.load_MSL_F7()
    elif dataset == "UCR_InternalBleeding16":
        training, test, valid = dp.load_UCR_InternalBleeding16()
    elif dataset == "UCR_InternalBleeding17":
        training, test, valid = dp.load_UCR_InternalBleeding17()

    train_dataset = AESegLoader(training, config.win_size, "train")
    valid_dataset = AESegLoader(valid, config.win_size, "valid")
    test_dataset = AESegLoader(test, config.win_size, "test")
    
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


    test_dataloader =  DataLoader(
        dataset = test_dataset, 
        batch_size = config.batch_size,
        shuffle=False,
        drop_last=True,
    )

    return train_dataloader, valid_dataloader, test_dataloader


# bae_ens = BAE_Ensemble(10, VAEModule, config)


def valid(bae_ens, data_loader):
    bae_ens.eval()
    total_loss = []
    with torch.no_grad():
        for i, x in enumerate(data_loader):
            x = x.view(config.batch_size, config.in_channel, -1)
            x = x.to(config.device)
            loss = bae_ens.fit(x).to(config.device)
            total_loss.append(loss.item())
    return np.average(total_loss)


def train(train_dataloader, valid_dataloader, bae_ens):

    bae_ens.train()
    for epoch in range(config.n_epochs):
        print(f"-------------- epoch: {epoch} ----------------")
        total_loss = []
        for i, x in enumerate(train_dataloader):
            bae_ens.zero_optimizers()
            x = x.view(config.batch_size, config.in_channel, -1) # [batch_size, in_channel, win_size]
            x = x.to(config.device)
            loss = bae_ens.fit(x)
            total_loss.append(loss.item())
            if (i + 1) % 20 == 0:
                print(f"     batch: {i + 1}, batch_loss: {loss.item()}     ")
            
            loss.backward()
            bae_ens.step_optimizers()
        train_loss = np.average(total_loss)
        valid_loss = valid(bae_ens, valid_dataloader)

        print(f">>>> Epoch: {epoch}, train_loss: {train_loss}, valid_loss: {valid_loss} <<<<")

def get_anomaly_score(test_dataloader, bae_ens):
    '''
        get the anomaly score(predict error) of the test data
    '''
    data_len = len(test_dataloader) * config.batch_size

    error = np.zeros((data_len, M))
    bae_ens.eval()
    for i, x in enumerate(test_dataloader):
        x = x.view(config.batch_size, config.in_channel, -1)
        x = x.to(config.device)
        predictions = bae_ens.predict_stack_values(x) # (10, batch_size, in_channel, win_size)

        for m in range(M):
            error[i*config.batch_size: (i+1)*config.batch_size, m] = torch.mean((x - predictions[m])**2, dim=(1,2)).cpu().detach().numpy()

    return error

def get_anomaly_threshold(bae_ens, train_dataloader):

    data_len = len(train_dataloader) * config.batch_size
    train_nll = np.zeros((data_len, M))

    bae_ens.eval()
    for i, x in enumerate(train_dataloader):
        x = x.view(config.batch_size, config.in_channel, -1)
        x = x.to(config.device)
        predictions = bae_ens.predict_stack_values(x)
        for m in range(M):
            train_nll[i*config.batch_size: (i+1)*config.batch_size, m] = torch.mean((x - predictions[m])**2, dim=(1,2)).detach().cpu().numpy()

    train_nll_mean = np.mean(train_nll, axis=1)
    threshold = np.percentile(train_nll_mean, 95)

    return threshold




def get_label(error_mean, threshold, dataset, seed):
    '''
        Get the gt and  predict label of the test data based on the threshold.\\
        And adjust the predict label based on the ground truth label. \\
    '''
    anomaly_index = get_anomaly_index(dataset)

    # smooth the reconstruction error
    filter_win = 30
    recons_error_smoothed = np.convolve(error_mean, np.ones(filter_win)/filter_win, mode='same')
    prefix = np.ones(config.win_size - 1) * recons_error_smoothed[0]
    recons_error_smoothed = np.concatenate((prefix, recons_error_smoothed))

    # construct predict label and adjust
    predict_label = np.zeros(recons_error_smoothed.shape)
    predict_label[recons_error_smoothed > threshold] = 1

    gt_label = np.zeros(recons_error_smoothed.shape)

    for index in anomaly_index:
        gt_label[index[0]: index[1]] = 1

    pred, gt = utils.adjuct_pred(predict_label, gt_label)
    pred, gt = utils.adjuct_gt(pred, gt)
    pred[-1] = 0 # prevent from array bounds
    gt[-1] = 0
    anomaly_index = utils.get_anomaly_segment(gt) 
    for index in anomaly_index:
        gt_label[index[0]: index[1]] = 1

    return gt, pred

def get_calibration_data(error, threshold, gt_label):
    '''
        Args:
        error shape: [seq_len, M]
        gt_label shape: [seq_len]
        threshold: float
    '''
    # expand error to the same shape as the original signal
    error_of_expand = []
    for i in range(M):
        prefix = np.ones(config.win_size-1) * error[0, i]
        error_of_expand.append(np.concatenate((prefix, error[:, i])))
    error_of_expand = np.array(error_of_expand)

    # smotth error
    filter_win = 30
    for m in range(M):
        error_of_expand[m] = np.convolve(error_of_expand[m], np.ones(filter_win)/filter_win, mode='same')

    # get the prediction probability of anomaly
    pred_prob = []
    for i in range(error_of_expand.shape[1]):
        pred_i = 0
        for m in range(M):
            if error_of_expand[m, i] > threshold:
                pred_i += 1
        pred_i = pred_i / M
        pred_prob.append(pred_i)
    pred_prob = np.array(pred_prob)
    # print(error_of_expand.shape, gt_label.shape, pred_prob.shape)
    split_probs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    # split_probs = np.linspace(0, 0.5, 10)

    calibration_data = []

    for prob in split_probs:
        obser_num = 0
        if prob == split_probs[0]:
            idx_s = np.where((pred_prob <= prob) & (pred_prob >= 0))[0]
        else:
            idx_s = np.where((pred_prob <= prob) & (pred_prob > round(prob - 0.1, 2)))[0]
        if len(idx_s) == 0:
            # continue
            calibration_data.append((prob, prob)) # if no data in the interval, the observed probability is equal to the predicted probability
            continue

        for i in idx_s:
            if gt_label[i] == 1:
                obser_num += 1
        calibration_data.append((prob, obser_num / len(idx_s)))
        
    calibration_data = np.array(calibration_data)
    print(calibration_data)
    return calibration_data

def save_result(calibration_data, dataset, loss_type, seed):
    dirname = f"./result/calibration"
    os.makedirs(dirname, exist_ok=True)
    if loss_type == 'mse':
        dataset =  dataset + "_epis" #"UCR-InternalBleeding17_Epis_Alea"
    elif loss_type == 'rnll':
        dataset = dataset + "_epis_alea_ours"
    else:
        dataset = dataset + "_epis_alea"

    df = pd.DataFrame(calibration_data, columns=['Predicted Probability', 'Observed Probability'])
    df.to_csv(f'{dirname}/{dataset}_seed{seed}.csv', index=False)

class Config:
    # model
    in_channel = 1
    win_size = 128
    latent_dim = 10
    hidden_num_units = 64

    
    # training
    n_epochs = 10
    lr = 1e-3
    batch_size = 64

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    train_dataloader, valid_dataloader, test_dataloader = get_dataloader(args.dataset)
    bae_ens = BAE_Ensemble(M, AutoEncoder, config)
    bae_ens.set_loss_type(args.loss_type)
    bae_ens.toDevice(config.device)

    # training
    config.n_epochs = 10
    train(train_dataloader, valid_dataloader, bae_ens)

    if args.loss_type != "mse" :
        config.n_epochs = args.epochs2
        print("-------------- var tunning --------------")
        bae_ens.tunning_var(args.learning_rate)
        bae_ens.set_loss_type(args.loss_type)
        bae_ens.set_rnll_hyper(args.lambda1, args.lambda2) # only for rnll loss
        train(train_dataloader, valid_dataloader, bae_ens)

    error = get_anomaly_score(test_dataloader, bae_ens)
    error_mean = np.mean(error, axis=1)

    threshold = get_anomaly_threshold(bae_ens, train_dataloader)
    gt_label, pred_label = get_label(error_mean, threshold, seed=args.seed, dataset=args.dataset)
    calibration_data = get_calibration_data(error, threshold, gt_label)

    print(f"----------- saving results --------------")
    save_result(calibration_data, args.dataset, args.loss_type, args.seed)


if __name__ == "__main__":
    args = get_args()

    config = Config()
    M = args.ensemble_examples
    config.win_size = args.win_size

    main(args)
