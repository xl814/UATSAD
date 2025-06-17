import torch
import torch.nn as nn
import torch.distributions as distributions
import numpy as np
import math


class AutoEncoder(torch.nn.Module):
    def __init__(self, config, homoscedastic=False):
        super().__init__()
        
        self.in_channel = config.in_channel
        self.latent_dim = config.latent_dim
        self.hidden_num_units = config.hidden_num_units
        self.win_size = config.win_size
        self.device = config.device

        # SensitiveHUE super-arguments
        self.alpha = 1

        self.weight_decay = 1e-2
        self.hidden_num_units_coeff = config.win_size // 8
        self.conv1 = torch.nn.Conv1d(in_channels=self.in_channel, out_channels=self.hidden_num_units // 8, kernel_size=4, stride=2, padding=1)
        self.conv2 = torch.nn.Conv1d(in_channels=self.hidden_num_units // 8, out_channels=self.hidden_num_units // 4, kernel_size=4, stride=2, padding=1)
        self.conv3 = torch.nn.Conv1d(in_channels=self.hidden_num_units // 4, out_channels=self.hidden_num_units, kernel_size=4, stride=2, padding=1)

        self.conv4 = nn.ConvTranspose1d(in_channels=self.hidden_num_units, out_channels=self.hidden_num_units // 4, kernel_size=4, stride=2, padding=1)
        self.conv5 = nn.ConvTranspose1d(in_channels=self.hidden_num_units // 4, out_channels=self.hidden_num_units // 8, kernel_size=4, stride=2, padding=1)
        self.conv6 = nn.ConvTranspose1d(in_channels=self.hidden_num_units // 8, out_channels=self.in_channel, kernel_size=4, stride=2, padding=1)
        # 1. encoder
        self.conv = nn.Sequential(
            self.conv1,
            nn.LeakyReLU(),
            self.conv2,
            nn.LeakyReLU(),
            self.conv3,
            nn.LeakyReLU()
        )
    
        self.fc1 = nn.Sequential(
            nn.Linear(in_features=self.hidden_num_units * self.hidden_num_units_coeff, out_features=self.latent_dim), # 8 = win_size // 8
        )    

        # 2. decoder
        self.fc2 = nn.Sequential(
            nn.Linear(in_features=self.latent_dim, out_features=self.hidden_num_units * self.hidden_num_units_coeff),
        ) 
        self.conv_decoder = nn.Sequential(
            self.conv4,
            nn.LeakyReLU(),
            self.conv5,
            nn.LeakyReLU(),
            self.conv6,
        )

        # 3. get mean and variance
        self.fc3 = nn.Sequential(
            nn.Linear(in_features = self.win_size, out_features = self.win_size),
        )

        self.log_var_net = nn.Sequential(
            nn.Linear(in_features = self.win_size, out_features= 16 * self.win_size),
            nn.Tanh(),
            nn.Linear(in_features = 16 * self.win_size,  out_features= 16 * self.win_size),
            nn.Tanh(),
            nn.Linear(in_features = 16 * self.win_size,  out_features= self.win_size),
            nn.Softplus()
        )

        self.mean_net = nn.ModuleList([
            self.conv1,
            self.conv2,
            self.conv3,
            self.conv4,
            self.conv5,
            self.conv6,
            self.fc1,
            self.fc2,
            self.fc3 ])
       
        self.first = False
        self.loss_type = 'mse'
        self.lambda1 = 2
        self.lambda2 = 1

    def encode(self, x):
        z = self.conv(x)
        z = z.view(z.size(0), -1)
        z = self.fc1(z)

        return z
    
    
    def decode(self, z):

        z = self.fc2(z)
        z = z.view(z.size(0), self.hidden_num_units, self.hidden_num_units_coeff)
        x_hat = self.conv_decoder(z)
        mean = self.fc3(x_hat)
        return mean
    
    def set_punishment(self, punishment):
        self.punishment = punishment

    def set_logvar(self, log_var):
        self.log_var = log_var
    
    def get_logvar(self):
        if self.log_var != None:
            return self.log_var

    # deprecated, not used !!!!! 
    def forward(self, x, alpha = None): 

        z = self.encode(x)
        x_hat = self.decode(z)

        loss = self.loss(x, x_hat)
        return x_hat, loss
    
    def predict(self, x):
        z = self.encode(x)
        x_hat = self.decode(z)
        log_var = self.log_var_net(x)
        return x_hat, log_var
        
    def loss(self, x, x_hat):
        loss = nn.MSELoss().to(self.device)
        return loss(x, x_hat)
    
    def loss_with_var(self, x, x_hat, log_var, lambda1 = 2, lambda2 = 1): 

        residual: torch.Tensor = (x - x_hat) ** 2
        var = log_var

        if self.loss_type == 'rnll':
            noise = (var.sqrt() * torch.randn_like(x_hat, device=self.device)).detach()
            avg_resi = torch.mean((x - x_hat - noise)).detach()
            regularizer = (lambda1 / (1 + (lambda1 - 1) * torch.exp(-lambda2 * ((x - x_hat - noise)**2 - avg_resi)) )).detach()         

            loss = torch.sum(residual / (var * regularizer) + torch.log(var)  + np.log(2*np.pi), dim=(1, 2)).to(self.device)

        elif self.loss_type == 'nll':
            loss = torch.sum(residual / var + torch.log(var) + np.log(2*np.pi), dim=(1, 2)).to(self.device)

        elif self.loss_type == 'beta_nll':
            beta_var = var.detach() ** 0.5 # beta = 0.5 can achieve the best trade-off between accuracy and log-likelihood.
            loss = torch.sum((
                    beta_var * (residual / (var) + torch.log(var)  + np.log(2*np.pi))
                ), dim=(1, 2))
            
        elif self.loss_type == 'mts_nll':
            mean_var = torch.mean(var, dim=(0, 1, 2)).detach() ** self.alpha
            beta_var = var.detach() ** 1 # beta = 1 in mts_nll
            loss = torch.sum((
                    beta_var * (residual / (var) + torch.log(var)  + np.log(2*np.pi))
                ), dim=(1, 2)) / mean_var
        else:
            loss = torch.sum((residual + np.log(2*np.pi)), dim=(1,2)).to(self.device)

        return torch.mean(loss)
    
    def loss_with_var_prior(self, x, x_hat):
        # suppose var is 1
        mse = torch.sum(((x - x_hat)**2 + np.log(2*np.pi)), dim=(1,2)).to(self.device)
        return torch.mean(mse)
    
    def fit(self, x):
        return self.criterion(x)
    
    def tunning(self):
        self.loss_type = 'rnll'
        for param in self.log_var_net.parameters():
            param.requires_grad = True

    def stop_var_weight_update(self):
        self.loss_type = 'mse'
        for name, param in self.log_var_net.named_parameters():
            param.requires_grad = False
            print(f"logvar net: Layer: {name}, requires_grad: {param.requires_grad}")

    def set_rnll_hyper(self, lambda1, lambda2):
        self.lambda1 = lambda1
        self.lambda2 = lambda2

    def set_loss_type(self, loss_type):
        self.loss_type = loss_type

    def criterion(self, x, batch):
        z = self.encode(x)
        x_hat = self.decode(z)
        log_var = self.log_var_net(x)

        loss = self.loss_with_var(x, x_hat, log_var=log_var, lambda1=self.lambda1, lambda2=self.lambda2)
        return loss 
    
    def log_prior_loss(self):
        weights, coeff = self.get_weights(device=self.device)
        prior_loss = torch.pow(coeff*(weights - self.anchored_prior), 2).sum().to(self.device)  
        return prior_loss
    