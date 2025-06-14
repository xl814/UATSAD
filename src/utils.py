import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import savefig
from statsmodels.distributions import ECDF

class ECDF_Wrapper(ECDF):
    def fit(self, x):
        return super(ECDF_Wrapper, self).__init__(x)

    def predict(self, x):
        return self(x)

    def cdf(self, x):
        return self(x)

class BAE_Outlier_proba:
    def __init__(self):
        self.dist_ = []
        self.norm_scaling_param = []
        
    def fit(self, training_scores):
        """
        fits multiple distrubutions on BAE anomaly scores of trainging data
        - `multi_scores`: list of np.array, shape=(n_ensemble, samples)
        """
        for i in range(len(training_scores[0])):
            # every score is a list of anomaly scores for each data point
            ecdf = ECDF_Wrapper(training_scores[:, i])
            proba = ecdf.cdf(training_scores[:, i])
            self.norm_scaling_param.append(np.mean(proba))
            self.dist_.append(ecdf)

    def predict(self, test_scores):
        """
        test_scores: np.array, shape=(n_samples, n_ensemble)
        """
        probas = []
        std = 0.3 
        sample_size = 20

        # for i in range(test_scores.shape[0]):
        #     samples = np.random.multivariate_normal(test_scores[:, i], np.eye(len(test_scores[:, i])) * sigma2, sample_size)
        #     samples = np.random.randn(sample_size, len(test_scores[:, i])) * std + test_scores[:, i]
        #     proba_i = []
        #     for sample in samples:
        #         proba = self.dist_[i].cdf(sample)
        #         proba = np.clip((proba - self.norm_scaling_param[i]) / (1 - self.norm_scaling_param[i]), 0, 1)
        #         proba_i.append(proba)
        #     proba = np.mean(proba_i, axis=0)
        #     # proba = np.clip((proba - self.norm_scaling_param[i]) / (1 - self.norm_scaling_param[i]), 0, 1)

        #     # proba = self.dist_[i].cdf(test_scores[:, i])
        #     # proba = np.clip((proba - self.norm_scaling_param[i]) / (1 - self.norm_scaling_param[i]), 0, 1)
        #     probas.append(proba)

        for m in range(test_scores.shape[1]):
            samples = []
            for i in range(sample_size):
                sample = np.random.randn(test_scores.shape[0]) * std + test_scores[:, m] # sample test_scores.shape[0] y-values(i.e. error value)
                samples.append(sample)
            proba_m = []
            for sample in samples:
                proba = self.dist_[m].cdf(sample) # sample's shape (test_scores.shape[0], )
                proba_m.append(proba)
            proba = np.mean(proba_m, axis=0)
            probas.append(proba)
        return np.array(probas)


def windows_to_sequences(seq_size, win_size, embeddings):
    """
        rolling window to non-overlapping windows sequences
        code reference: https://github.com/lin-shuyu/VAE-LSTM-for-anomaly-detection/blob/master/codes/data_loader.py
        return: np.array, shape=(n_seqs, seq_size, win_size, latent_dim)
        
        Args:
        - `data`: np.array, shape=(n_samples, latent_dim)
    """
    n_samples = embeddings.shape[0] + win_size - 1
    latent_dim = embeddings.shape[1]
    for k in range(win_size):
        n_not_overlap_wins = (n_samples - k) // win_size
        n_seqs = n_not_overlap_wins - seq_size + 1
        sequences = np.zeros((n_seqs, seq_size, latent_dim))
        for i in range(n_seqs):
            cur_seq = np.zeros((seq_size, latent_dim))
            for j in range(seq_size):
                # print(f"i: {i}, j: {j}, k: {k}, win_size: {win_size}")
                cur_seq[j] = embeddings[k + win_size * (i + j)]
            sequences[i] = cur_seq
        if k == 0:
            lstm_seq = sequences
        else:
            lstm_seq = np.concatenate((lstm_seq, sequences), axis=0)
    print(f"lstm_seq shape: {lstm_seq.shape}")
    return lstm_seq

def produce_embeddings(vae_model, data_loader, code_size, batch_size):
    """
    produce embeddings from VAE model
    Args:
    - `data`: np.array, shape=(n_wins, n_features)
    """
    embedding_lstm = np.zeros((len(data_loader) * batch_size, code_size))
    for i, (x, _) in enumerate(data_loader):
        x = x.view(-1, x.shape[2], x.shape[1])
        _, mu, _ = vae_model.encode(x)
        embedding_lstm[i * batch_size: (i + 1) * batch_size] = mu.detach().numpy()
    print(f"embedding_lstm shape: {embedding_lstm.shape}")
    return embedding_lstm

def decode_embeddings(vae_model, data_loader, n_feature, batch_size):
    """
    produce reconstructed signals from VAE model.

    Args:
    - `data_loader`: DataLoader
    - `n_feature`: int, number of features
    """
    embedding_lstm_decoded = np.zeros((len(data_loader) * batch_size, n_feature))
    for i, x in enumerate(data_loader):
        x_hat = vae_model.decode(x)
        print(f"x_hat shape: {x_hat.shape}")
        x_hat = x_hat.view(x_hat.shape[0], -1)
        embedding_lstm_decoded[i * batch_size: (i + 1) * batch_size] = x_hat.detach().numpy()
    
    print(f"embedding_lstm_decoded shape: {embedding_lstm_decoded.shape}")
    return embedding_lstm_decoded

def plot_reconstructed_signal(original_signals, reconstructed_signal, dim):
    n_signals = original_signals.shape[0]
    for i in range(dim):
        fig, axs = plt.subplots(2, 5, figsize=(18, 10), edgecolor='k')
        fig.subplots_adjust(hspace = .5, wspace=.01)
        axs = axs.ravel()
        for j in range(n_signals):
            axs[j].plot(original_signals[j, i, :], label="original")
            axs[j].plot(reconstructed_signal[j, i, :], label="reconstructed")
            axs[j].grid(True)
            axs[j].set_xlim(0, original_signals.shape[2])
            axs[j].set_ylim(-2, 2)
            if j == 0:
                axs[j].legend()
        plt.suptitle(f"Dimension {i}")
        savefig(f"test_reconstructed_dim{i}_test_ensemble.pdf")
        fig.clf()
        plt.close(fig)

def plot_reconstructed_confidence_signal(upper_bound, lower_bound, original_signals, dim):
    n_signals = original_signals.shape[0]
    for i in range(dim):
        fig, axs = plt.subplots(2, 5, figsize=(18, 10), edgecolor='k')
        fig.subplots_adjust(hspace = .5, wspace=.01)
        axs = axs.ravel()
        for j in range(n_signals):
            axs[j].plot(original_signals[j, i, :], label="original")
            axs[j].plot(upper_bound[j, i, :], label="upper bound")
            axs[j].plot(lower_bound[j, i, :], label="lower bound")
            axs[j].fill_between(np.arange(original_signals.shape[2]), upper_bound[j, i, :], lower_bound[j, i, :], color='gray', alpha=0.5)
            axs[j].grid(True)
            axs[j].set_xlim(0, original_signals.shape[2])
            axs[j].set_ylim(-2, 2)
            if j == 0:
                axs[j].legend()
        plt.suptitle(f"Dimension {i}")
        savefig(f"test_reconstructed_confidence_dim{i}_ensemble.pdf")
        fig.clf()
        plt.close(fig)

def plot_pred_embeddings(embeddings, pred_embeddings, dim):
    n_embeddings = embeddings.shape[0]
    fig, axs = plt.subplots(2, 5, figsize=(18, 10), edgecolor='k')
    fig.subplots_adjust(hspace = .5, wspace=.01)
    axs = axs.ravel()
    for i in range(n_embeddings):
        axs[i].plot(embeddings[i, ], label="original")
        axs[i].plot(pred_embeddings[i, ], label="predicted")
        axs[i].grid(True)
        axs[i].set_xlim(0, embeddings.shape[0])
        axs[i].set_ylim(-2, 2)
        if i == 0:
            axs[i].legend()
    plt.suptitle("Predicted Embeddings")
    savefig(f"test_pred_embeddings.pdf")
    fig.clf()
    plt.close(fig)


def ecdf(train_data, x):
    """
    Empirical Cumulative Distribution Function
    - `train_data`: np.array, shape=(n_samples,)
    - `x`: float or np.array, shape=(n_samples,)
    """
    n = train_data.shape[0]
    return np.searchsorted(np.sort(train_data), x) / n


def get_anomaly_segment(gt):
    """
    get anomaly segment from ground truth labels
    - `gt`: np.array, shape=(n_samples,)
    """
    gt[-1] = 0
    anomaly_index = []
    diff = np.diff(gt)
    # print(diff.shape)
    start = np.where(diff == 1)[0] + 1
    end = np.where(diff == -1)[0]
    # print(start, end)
    for i in range(len(start)):
        anomaly_index.append((start[i], end[i]))
    # print(np.where(diff == 1)[0])
    return anomaly_index

def adjuct_gt(pred, gt):
    anomaly_state = False
    for i in range(len(gt)):
        if gt[i] == 1 and pred[i] == 1: # and not anomaly_state:
            anomaly_state = True
            for j in range(i, 0, -1):
                if pred[j] == 0:
                    break
                else:
                    if gt[j] == 0:
                        gt[j] = 1
            for j in range(i, len(gt)):
                if pred[j] == 0:
                    break
                else:
                    if gt[j] == 0:
                        gt[j] = 1
        # elif gt[i] == 0:
        #     anomaly_state = False
        # if anomaly_state:
        #     pred[i] = 1
    return pred, gt

def adjuct_pred(pred, gt):
    """
    Adjust predicted labels based on the ground truth labels
    - `pred_label`: np.array, shape=(n_samples,)
    - `gd_label`: np.array, shape=(n_samples,)

    ref: `Unsupervised Anomaly Detection via Variational Auto-Encoder for Seasonal KPIs in Web Applications`
    """
    anomaly_state = False
    for i in range(len(gt)):
        if gt[i] == 1 and pred[i] == 1: #and not anomaly_state:
            # anomaly_state = True
            for j in range(i, 0, -1):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
            for j in range(i, len(gt)):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
        # elif gt[i] == 0:
        #     anomaly_state = False
        # if anomaly_state:
        #     pred[i] = 1
    return pred, gt

def gaussian_function(x, mu, var):
    """
    Gaussian function (Homosedasticity)
    - `x`: np.array, shape=(n_samples,)
    - `mu`
    - `sigma`: float
    """
    return np.exp(-0.5 * ((x - mu)**2 / 2 * var)) / np.sqrt(2 * np.pi * var)

def differential_entropy( var):
    return np.log(2 * np.pi * np.e * var) / 2


def generate_adversarial_sample(x, win_size, sample_num = 20):
    '''
        x: np.array, shape=(n_samples, win_size, n_features)
    '''
    max_epsilon = win_size * 0.01
    sigma = 0.3
    # uniform noise from -max_epsilon to max_epsilon
    adversial_samples = []
    for _ in range(sample_num):
        # noise = np.random.rand(*x.shape) * 2 * max_epsilon - max_epsilon
        noise = np.random.normal(0, sigma, x.shape)
        x_adv = x + noise
        adversial_samples.append(x_adv) 

    return np.array(adversial_samples)

import torch 

class Synthetic_Data_generator():
    def __init__(self, std_x_noise, std_y_noise, rho_x, mu_x):
        self.std_x_noise = std_x_noise
        self.std_y_noise = std_y_noise
        self.rho_x = rho_x
        self.mu_x = mu_x
        

    def get_y_std(self):
        return self.std_y_noise

    def get_x_std(self):
        return self.std_x_noise

    def x_to_y(self,x):
        return x * (1 + torch.sin(x))
    
    def get_data(self, total_examples, device = 'gpu'):

        x = torch.linspace(self.mu_x - 2 * self.rho_x, self.mu_x + 2 * self.rho_x, total_examples).unsqueeze(-1)
        # x = x.to(device)
        y = self.x_to_y(x)

        x_tilde = x + torch.randn(x.shape) * self.get_x_std()
        y_tilde = y + torch.randn(y.shape) * self.get_y_std()

        data = {'x':x,'y':y,'x_tilde':x_tilde,'y_tilde': y_tilde}

        return data

class Synthetic_heteroscedasticData_generator():

    def __init__(self,  std_y_noise, rho_x, mu_x, anomaly_value):
        # self.std_x_noise = std_x_noise
        self.std_y_noise = std_y_noise
        self.rho_x = rho_x
        self.mu_x = mu_x
        self.anomaly_value = anomaly_value

    def get_y_std(self , x):
        return self.std_y_noise * (1 + 0.1 * x)
        # return self.std_y_noise * (1 + 3 * torch.cos(x) + 7 * torch.sin(x))

    def get_y_std_another(self, x):
        return self.std_y_noise * torch.abs(torch.cos(x/2))

    # def get_x_std(self , x):
    #     return self.std_x_noise

    def x_to_y(self,x):
        return x * (1 + torch.sin(x))
    
    def set_anomaly_value(self, value):
        self.anomaly_value = value

    def set_std_y_noise(self, value):
        self.std_y_noise = value
    
    def get_data(self, total_examples):

        x = torch.linspace(self.mu_x - 2 * self.rho_x, self.mu_x + 2 * self.rho_x, total_examples).unsqueeze(-1)
        # y = x*a
        y = self.x_to_y(x)
        y_disterb = self.x_to_y(x)
        y_disterb[200: 250] = self.anomaly_value # -4 -11
        
        # x_tilde = x + torch.randn(size=x.shape) * self.get_x_std(x)
        y_tilde = y + torch.randn(size=y.shape) * self.get_y_std(y) # get_y_std(x)
        y_disterb_tilde = y_disterb + torch.randn(size=y_disterb.shape) * self.get_y_std(y_disterb)
        data = {'y_disturb':y_disterb,'y':y, 'y_disturb_tilde':y_disterb_tilde, 'y_tilde': y_tilde}

        return data
    
def remove_outliers_iqr(data, factor=1.5):
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1
    lower = q1 - factor * iqr
    upper = q3 + factor * iqr
    
    data = np.clip(data, a_min=None, a_max=upper)
    return data

def smooth(data: np.ndarray, filter_win: int = 10) -> np.ndarray:
    filter_win = filter_win
    data_smoot = np.convolve(data, np.ones(filter_win)/filter_win, mode='same')
    return data_smoot


def tunning_for_ensemble(data_loader, bvae_ens, config, optim, log_var_net, tunning_win = 32):
    bvae_ens.train()
    for epoch in range(config.n_epochs):
        print(f"--------------tunning epoch: {epoch} ----------------")
        total_loss = []
        for i, x in enumerate(data_loader):
            optim.zero_grad()
            loss = 0
            x = x.view(config.batch_size, config.in_channel, -1) # (batch_size, in_channel, win_size)

            x.requires_grad = False
            x_pred = bvae_ens.predict_mean(x)
            
            # log_var_approx = log_var_net(x.unsqueeze(-1)).squeeze(-1)
            for k in range(config.win_size // tunning_win):
                x_win = x[:, :, k*tunning_win: (k+1)*tunning_win]
                x_pred_win = x_pred[:, :, k*tunning_win: (k+1)*tunning_win]
                log_var_approx = log_var_net(x_win)
            # log_var_approx = log_var_approx.view(config.batch_size, config.in_channel, -1)
                loss += torch.sum((torch.exp(-log_var_approx) * (x_win - x_pred_win)**2 + log_var_approx + np.log(2*np.pi)), dim=(1,2))
            loss = loss.mean()
            total_loss.append(loss.item())
            if (i + 1) % 10 == 0:
                print(f"     batch: {i + 1}, batch_loss: {loss.item()}     ")
            loss.backward()
            optim.step()
        train_loss = np.average(total_loss)
        print(f">>>> Epoch: {epoch}, train_loss: {train_loss} <<<<")


def adjust_learning_rate(optimizer, epoch, lr):
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    lr_adjust = {epoch: lr * (0.5 ** ((epoch - 1) // 1))}
    for param_group in optimizer.param_groups:
            print(param_group)
    return lr

def logging(log_name, dataset, calibrate_epic, calibrate_epis_alea):
    file = open("./logs/"+log_name, "a")  # append mode
    with open("./logs/"+log_name, "r") as file:
        lines = file.readlines()
        if len(lines) == 0:
            file.write("dataset, calibrate_epic, calibrate_epis_alea\n")
    message = f"dataset: {dataset}, calibrate_epic: {calibrate_epic}, calibrate_epis_alea: {calibrate_epis_alea}\n"
    file.write(message)
    print(message)