import torch
import torch.nn as nn
import torch.distributions as distributions
import numpy as np
import math

class VAEModule(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.in_channel = config.in_channel
        self.latent_dim = config.latent_dim
        self.hidden_num_units = config.hidden_num_units
        self.win_size = config.win_size
        self.sigma2 = config.sigma ** 2
        
        self.weight_decay = 1e-1
        self.hidden_num_units_coeff = config.win_size // 8
        self.conv1 = torch.nn.Conv1d(in_channels=self.in_channel, out_channels=self.hidden_num_units // 16, kernel_size=3, stride=2, padding=1)
        self.conv2 = torch.nn.Conv1d(in_channels=self.hidden_num_units // 16, out_channels=self.hidden_num_units // 4, kernel_size=3, stride=2, padding=1)
        self.conv3 = torch.nn.Conv1d(in_channels=self.hidden_num_units // 4, out_channels=self.hidden_num_units, kernel_size=3, stride=2, padding=1)

        self.conv4 = nn.ConvTranspose1d(in_channels=self.hidden_num_units, out_channels=self.hidden_num_units // 4, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv5 = nn.ConvTranspose1d(in_channels=self.hidden_num_units // 4, out_channels=self.hidden_num_units // 16, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv6 = nn.ConvTranspose1d(in_channels=self.hidden_num_units // 16, out_channels=self.in_channel, kernel_size=3, stride=2, padding=1, output_padding=1)
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
            nn.Linear(in_features=self.hidden_num_units * self.hidden_num_units_coeff, out_features=self.latent_dim * 4), # 8 = win_size // 8
            nn.LeakyReLU()
            )    
        self.fc_mu = nn.Linear(in_features=self.latent_dim * 4, out_features=self.latent_dim)
        self.fc_std = nn.Sequential(
            nn.Linear(in_features=self.latent_dim * 4, out_features=self.latent_dim),
            nn.ReLU()
        )

        # 2. decoder
        self.fc2 = nn.Sequential(
            nn.Linear(in_features=self.latent_dim, out_features=self.hidden_num_units * self.hidden_num_units_coeff),
            nn.LeakyReLU()
        ) 
        self.conv_decoder = nn.Sequential(
            self.conv4,
            nn.LeakyReLU(),
            self.conv5,
            nn.LeakyReLU(),
            self.conv6,
        )
        
        # init anchor weight
        self.init_anchored_weight()

    def encode(self, x):
        x = self.conv(x) # (batch_size, hidden_num_units, win_size)
        # print(x.size())
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        mu = self.fc_mu(x) # (batch_size, latent_dim)
        std = self.fc_std(x) + 1e-2 # (batch_size, latent_dim)
        
        # reparametrization trick
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z, mu, std
    
    def decode(self, z):
        z = self.fc2(z)
        z = z.view(z.size(0), self.hidden_num_units, self.hidden_num_units_coeff)
        x_hat = self.conv_decoder(z)
        return x_hat
        # x = x.view(z.size(0), z.size(1), 

    def forward(self, x, alpha = None):
        z, mu, std = self.encode(x)
        # print("after encode:", z.size())
        x_hat = self.decode(z)
        # print("after decode:", x_hat.size())

        if alpha is not None:
            beta = torch.zeros(alpha.size(0))
            torch.sum(alpha, dim=1, out=beta)
            beta = beta / self.win_size
            loss = self.m_elbo(alpha, beta, x, x_hat, z, mu, std**2, self.sigma2)
        else:
            loss = self.loss(x, x_hat, mu, std**2, self.sigma2)
        return x_hat, loss
    
    def fit(self, x):
        return self.criterion(x)
    
    def predict(self, x):
        z, mu, std = self.encode(x)
        x_hat = self.decode(z)
        return x_hat
        
    def loss(self, x, x_hat, z_mu, z_var, sigma2):
        kl_loss =torch.sum( 0.5 * (z_mu ** 2 + z_var - torch.log(z_var) - 1), dim=1) 
        # print("kl_loss:", z_var, )
        kl_loss = kl_loss.mean()
        rec_loss = torch.sum((x_hat - x) ** 2 / (2 * sigma2), dim=(1, 2)) + self.win_size * np.log(2 * torch.pi * sigma2) * 0.5
        rec_loss = rec_loss.mean()
        # print("rec_loss:", rec_loss)
        elbo_loss = rec_loss + kl_loss
        return elbo_loss
    
    def m_elbo(self, alpha, beta, x, x_hat, z, mu_z, var_z,  sigma2):
        """
            m_elbo loss from `Donut` paper

            Args:
            - `alpha`: a config.win_size dimensional vector
            - `beta`: a scalar
        """
        weighted_pxz = torch.zeros(z.size(0))
        weighted_pz = torch.zeros(z.size(0))
    
        pxz = torch.sum(((x_hat - x) ** 2) / (2 * sigma2), dim=1) + np.log(2 * torch.pi * sigma2) * 0.5
        for i in range(pxz.size(0)):
            weighted_pxz[i] = torch.dot(alpha[i], pxz[i])
        pz = torch.sum(z ** 2, dim=1) * 0.5 +  self.win_size * np.log(2 * torch.pi) * 0.5
        for i in range(pz.size(0)):
            weighted_pz[i] = beta[i] * pz[i]
        qzx = torch.sum((z-mu_z) ** 2 / (2 * var_z), dim=1) + self.win_size * np.log(2 * torch.pi) * 0.5 + 0.5 * torch.sum(torch.log(var_z), dim=1)
        
        weighted_pxz = weighted_pxz.mean()
        weighted_pz = weighted_pz.mean()
        qzx = qzx.mean()
        m_elbo = weighted_pxz + weighted_pz - qzx
        return m_elbo
    
    def criterion(self, x):
        z, mu, std = self.encode(x)
        x_hat = self.decode(z)
        loss1 = self.loss(x, x_hat, mu, std**2, self.sigma2)
        loss2 = self.log_prior_loss() + torch.sum((x - x_hat) ** 2, dim=(1, 2)).mean()
        return loss1 # + loss2 / x.shape[0]
    
    def log_prior_loss(self):
        weights = torch.cat([parameter.flatten() for parameter in self.parameters()])
        prior_loss = torch.pow((weights - self.anchored_prior), 2).sum() * self.weight_decay
        return prior_loss
    
    def init_anchored_weight(self, anchored_weight_mu = 0, anchored_weight_std = 1):
        weights = torch.cat([parameter.flatten() for parameter in self.parameters()])
        mu = torch.ones(weights.size()) * anchored_weight_mu
        init_weight = torch.randn(weights.size())
        self.anchored_prior = weights.detach().clone()
        # self.anchored_prior = init_weight

    def score(self, x):
        # mc sampling
        # batch_size should be 1
        scores = []
        for _ in range(100):
            z, mu, std = self.encode(x)
            x_hat = self.decode(z)
            rec_loss = torch.sum((x_hat - x) ** 2 / (2 * self.sigma2), dim=(1, 2)) + self.win_size * np.log(2 * torch.pi * self.sigma2) * 0.5
            scores.append(rec_loss.detach().numpy())
        
        scores = np.array(scores)
        scores = np.mean(scores, axis=0)
        return scores


class LSTMModule(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.lstm = nn.LSTM(input_size=config.in_channel, 
                            hidden_size=config.hidden_num_units_lstm, 
                            num_layers=config.num_layers_lstm, 
                            batch_first=True )
        
        self.fc = nn.Linear(in_features=config.hidden_num_units_lstm, out_features=config.in_channel)
        # self.lstm2 = nn.LSTM(input_size=config.hidden_num_units_lstm,
        #                     hidden_size=config.latent_dim,
        #                     num_layers=1,
        #                     batch_first=True)

    def forward(self, x, y):
        # print(x.size(), y.size())
        out, (_, _) = self.lstm(x)
        out = self.fc(out)
        # out, (_, _) = self.lstm2(out)
        # print(out.size(), x.size())
        loss = self.loss(y, out)
        return out, loss
    
    def predict(self, x):
        out, (_, _) = self.lstm(x)
        out = self.fc(out)
        return out
    
    def fit(self, x):
        out, (_, _) = self.lstm(x)
        out = self.fc(out)
        # mse = torch.sum(((x - out)**2 + np.log(2*np.pi)), dim=(1,2)).mean()
        mse = nn.MSELoss() 

        return mse(x, out)

class AutoEncoder(torch.nn.Module):
    def __init__(self, config, homoscedastic=False):
        super().__init__()
        
        self.in_channel = config.in_channel
        self.latent_dim = config.latent_dim
        self.hidden_num_units = config.hidden_num_units
        self.win_size = config.win_size
        self.sigma2 = config.sigma ** 2
        self.device = config.device

        self.eps = 0.15

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
            # nn.Linear(in_features= 4 * self.win_size, out_features= self.win_size),
            # nn.Tanh(),
            # nn.Linear(in_features=4 * self.win_size, out_features= 4 * self.win_size),
            # nn.Tanh(),
            # nn.Linear(in_features= 4 * self.win_size, out_features= self.win_size),
            # nn.LeakyReLU(),
            # nn.Linear(in_features= 2 * self.win_size, out_features=self.win_size),
        )

        self.log_var_net = nn.Sequential(
            nn.Linear(in_features = self.win_size, out_features= 16 * self.win_size),
            nn.Tanh(),
            # nn.Linear(in_features= 8 * self.win_size, out_features= 8 * self.win_size),
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
       
        
        # self.init_anchored_weight()
        # self.init_weight()
        # self.stop_var_weight_update()
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
            # sample_var = torch.var(x_hat, dim=(1, 2))
            # noise = (var.sqrt() * torch.randn_like(x_hat, device=self.device))
            # loss_item = (((torch.var(noise, dim=(1, 2)) - sample_var))**2).to(self.device)
            noise = (var.sqrt() * torch.randn_like(x_hat, device=self.device)).detach()
            # avg_resi = torch.mean((x - x_hat - noise)).detach()
            avg_resi = torch.mean((x - x_hat - noise)).detach()
            regularizer = (lambda1 / (1 + (lambda1 - 1) * torch.exp(-lambda2 * ((x - x_hat - noise)**2 - avg_resi)) # 3 2 3 
                                )).detach()         
            # regularizer = (2 / (1 + torch.exp(3 * (x - x_hat - noise)**2))).detach()
            # regularizer = torch.exp(-((x - x_hat - noise)**2) / 2)
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

        loss1 = self.loss_with_var(x, x_hat, log_var=log_var, lambda1=self.lambda1, lambda2=self.lambda2)

        # x_adv = self.adversarial_sample(x, loss1)
        # if x_adv == None:
        #     loss2 = 0 # x.grad = false 
        # else:
        #     # print("adversarial sampling ...")
        #     z_adv = self.encode(x_adv)
        #     x_hat_adv, log_var = self.decode(z_adv)
        #     # loss2 = self.loss(x_adv, x_hat_adv)
        #     loss2 = self.loss_with_var(x_adv, x_hat_adv, log_var)
        # loss3 = self.log_prior_loss() 

        return loss1 # + loss3 / x.shape[0]# + loss2 + loss3 / x.shape[0]
    
    def log_prior_loss(self):
        # weights = torch.cat([parameter.flatten() for parameter in self.parameters()])
        weights, coeff = self.get_weights(device=self.device)
        prior_loss = torch.pow(coeff*(weights - self.anchored_prior), 2).sum().to(self.device)  #* self.weight_decay
        # prior_loss = torch.pow((weights - self.anchored_prior), 2).sum().to(self.device) * self.weight_decay
        return prior_loss
    
    def get_weights(self, device = 'cpu'):
        '''
            Ignore bias
        '''
        weights = torch.tensor([], device=device)
        coeff = torch.tensor([], device=device)
        for name, param in self.named_parameters():
            if "weight" in name and ( "conv" in name or "fc" in name): 
                # print(weights.is_cuda, param.is_cuda, self.device)
                weights = torch.cat([weights, param.flatten()])
                coeff = torch.cat([coeff, self.weight_coef[name] * torch.ones_like(param.flatten())])
        return weights, coeff
    
    def init_weight(self):
        # for name, param in self.named_parameters():
        #     print(f"{name}: {param.size()}")
        for name, param in self.named_parameters():
            # print(f"Layer: {name}, param: {param.size()}")
            if "conv" in name and "weight" in name:
                fan_in = param.size()[1] * param.size()[2]
                std = 1 / math.sqrt(fan_in)  
                nn.init.normal_(param, 0, std)

            elif "fc" in name and "weight" in name:
                fan_in = param.size()[1]
                std = 1 / math.sqrt(fan_in)  
                nn.init.normal_(param, 0, std)

    def init_anchored_weight(self, anchored_weight_mu = 0, anchored_weight_std = 1):
        init_weights = torch.tensor([])
        self.weight_coef = {}
        for name, param in self.named_parameters():
            # print(f"Layer: {name}, param: {param.size()}")
            if "conv" in name and "weight" in name:
                fan_in = param.size()[1] * param.size()[2]
                std = 1 / math.sqrt(fan_in)  
                self.weight_coef[name] = 1 / std
                anchored = torch.randn_like(param) * std
                init_weights = torch.cat([init_weights, anchored.flatten()])
            elif "fc" in name and "weight" in name:
                fan_in = param.size()[1]
                std = 1 / math.sqrt(fan_in)  
                self.weight_coef[name] = 1 / std
                anchored = torch.randn_like(param) * std
                init_weights = torch.cat([init_weights, anchored.flatten()])

        init_weights, _ = self.get_weights()              
        # init_weights = torch.cat([parameter.flatten() for parameter in self.parameters()])
        self.anchored_prior = init_weights.detach().clone().to(self.device) # anchored weight not participate in gradient update
        # self.anchored_prior = init_weight


    def adversarial_sample(self, x, loss):
        if x.requires_grad == False:
            return None
        loss.backward(retain_graph=True)
        x_adv = x + 0.5 * torch.sign(x.grad)
        return x_adv
    

    

class Vanile_ae(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.in_channel = config.in_channel
        self.latent_dim = config.latent_dim
        self.hidden_num_units = config.hidden_num_units
        self.win_size = config.win_size
        self.sigma2 = config.sigma ** 2

        self.weight_decay = 1e-2
        self.hidden_num_units_coeff = config.win_size // 8
        self.conv1 = torch.nn.Conv1d(in_channels=self.in_channel, out_channels=self.hidden_num_units // 16, kernel_size=3, stride=2, padding=1)
        self.conv2 = torch.nn.Conv1d(in_channels=self.hidden_num_units // 16, out_channels=self.hidden_num_units // 4, kernel_size=3, stride=2, padding=1)
        self.conv3 = torch.nn.Conv1d(in_channels=self.hidden_num_units // 4, out_channels=self.hidden_num_units, kernel_size=3, stride=2, padding=1)

        self.conv4 = nn.ConvTranspose1d(in_channels=self.hidden_num_units, out_channels=self.hidden_num_units // 4, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv5 = nn.ConvTranspose1d(in_channels=self.hidden_num_units // 4, out_channels=self.hidden_num_units // 16, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv6 = nn.ConvTranspose1d(in_channels=self.hidden_num_units // 16, out_channels=self.in_channel, kernel_size=3, stride=2, padding=1, output_padding=1)
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

    def encode(self, x):

        z = self.conv(x)
        z = z.view(z.size(0), -1)
        z = self.fc1(z)

        return z
    def decode(self, z):
        z = self.fc2(z)
        z = z.view(z.size(0), self.hidden_num_units, self.hidden_num_units_coeff)
        x_hat = self.conv_decoder(z)
        return x_hat

    def forward(self, x, alpha = None):

        z = self.encode(x)
        x_hat = self.decode(z)

        loss = self.loss(x, x_hat)
        return x_hat, loss
    
    def predict(self, x):
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat
        
    def loss(self, x, x_hat):
        loss = nn.MSELoss()
        return loss(x, x_hat)
    
    def fit(self, x):
        return self.criterion(x)

    def criterion(self, x):
        z = self.encode(x)
        x_hat = self.decode(z)
        loss1 = self.loss(x, x_hat)
        return loss1  
    


class AutoEncoder_Droupout(torch.nn.Module):
    def __init__(self, config):
        super().__init__()

        self.p_drop = 0.2
        self.dropout = nn.Dropout(p=self.p_drop)
        
        self.in_channel = config.in_channel
        self.latent_dim = config.latent_dim
        self.hidden_num_units = config.hidden_num_units
        self.win_size = config.win_size
        self.sigma2 = config.sigma ** 2
        self.weight_decay = 1e-4

        self.hidden_num_units_coeff = config.win_size // 8
        self.conv1 = torch.nn.Conv1d(in_channels=self.in_channel, out_channels=self.hidden_num_units // 16, kernel_size=3, stride=2, padding=1)
        self.conv2 = torch.nn.Conv1d(in_channels=self.hidden_num_units // 16, out_channels=self.hidden_num_units // 4, kernel_size=3, stride=2, padding=1)
        self.conv3 = torch.nn.Conv1d(in_channels=self.hidden_num_units // 4, out_channels=self.hidden_num_units, kernel_size=3, stride=2, padding=1)

        self.conv4 = nn.ConvTranspose1d(in_channels=self.hidden_num_units, out_channels=self.hidden_num_units // 4, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv5 = nn.ConvTranspose1d(in_channels=self.hidden_num_units // 4, out_channels=self.hidden_num_units // 16, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv6 = nn.ConvTranspose1d(in_channels=self.hidden_num_units // 16, out_channels=self.in_channel, kernel_size=3, stride=2, padding=1, output_padding=1)
        # 1. encoder
        self.conv = nn.Sequential(
            self.conv1,
            nn.LeakyReLU(),
            self.dropout,
            self.conv2,
            nn.LeakyReLU(),
            self.dropout,
            self.conv3,
            nn.LeakyReLU(),
            self.dropout,
        )
    
        self.fc1 = nn.Sequential(
            nn.Linear(in_features=self.hidden_num_units * self.hidden_num_units_coeff, out_features=self.latent_dim), # 8 = win_size // 8
            self.dropout
        )    

        # 2. decoder
        self.fc2 = nn.Sequential(
            nn.Linear(in_features=self.latent_dim, out_features=self.hidden_num_units * self.hidden_num_units_coeff),
            self.dropout,
        ) 
        self.conv_decoder = nn.Sequential(
            self.conv4,
            nn.LeakyReLU(),
            self.dropout,
            self.conv5,
            nn.LeakyReLU(),
            self.dropout,
            self.conv6,
            
            # # added for following
            nn.LeakyReLU(),
            self.dropout
        )

        # 3. get mean and variance
        self.fc3 = nn.Sequential(
            nn.Linear(in_features=self.win_size, out_features=self.win_size),
        )
        self.fc4 = nn.Sequential(
            nn.Linear(in_features=self.win_size, out_features= 4 * self.win_size),
            nn.LeakyReLU(),
            nn.Linear(in_features=4 * self.win_size, out_features=self.win_size),
            #nn.LeakyReLU()
        )

        # fc4: stop weight update 

        self.stop_var_weight_update()
        # init anchor weight
        # self.init_weight()
        self.init_hyperparameters()
        self.first = False # wethere tunning variance

    def encode(self, x):
        # x = self.dropout(x)
        z = self.conv(x)
        z = z.view(z.size(0), -1)
        z = self.fc1(z)
        return z
    
    def decode(self, z):
        # z = self.dropout(z)
        z = self.fc2(z)
        z = z.view(z.size(0), self.hidden_num_units, self.hidden_num_units_coeff)
        x_hat = self.conv_decoder(z)
        
        # added for variance
        x_hat = self.fc3(x_hat)
        log_var = self.fc4(x_hat)
        return x_hat, log_var
    
    def var_net(self, x):
        log_var = self.fc4(x)
        return log_var

    def forward(self, x, alpha = None):

        z = self.encode(x)
        x_hat = self.decode(z)

        loss = self.loss(x, x_hat)
        return x_hat, loss
    
    def predict(self, x):
        z = self.encode(x)
        x_hat, log_var = self.decode(z)
        return x_hat, log_var
        
    def loss(self, x, x_hat):
        loss = nn.MSELoss()
        return loss(x, x_hat)
    
    def loss_with_var(self, x, x_hat, log_var, lambda1=0.3, lambda2=0.7):
        mse = torch.sum((torch.exp(-log_var) * (x - x_hat)**2 + log_var + np.log(2*np.pi)), dim=(1,2))
        return torch.mean(mse)
    
    def loss_with_var_prior(self, x, x_hat):
        mse = torch.sum(((x - x_hat)**2 + np.log(2*np.pi)), dim=(1,2))
        return torch.mean(mse)
    
    def fit(self, x):
        return self.criterion(x)

    def tunning(self):
        self.first = False
        for param in self.fc4.parameters():
            param.requires_grad = True

    def criterion(self, x, var=None):
        z = self.encode(x)
        mean, log_var = self.decode(z)
        # loss1 = self.loss(x, mean)   # note: this may need be reused
        if self.first:
            loss1 = self.loss_with_var_prior(x, mean)
        else:
            loss1 = self.loss_with_var(x, mean, log_var)


        loss2 = self.log_prior_loss() # + torch.sum((x - x_hat) ** 2, dim=(1, 2)).mean()

        x_adv = self.adversarial_sample(x, loss1)
        if x_adv == None:
            loss3 = 0 # x.grad = false 
        else:
            z_adv = self.encode(x_adv)
            x_hat_adv = self.decode(z_adv)
            loss3 = self.loss(x_adv, x_hat_adv)

        loss3 = 0 # cancel adversarial sample
        # print("loss1", loss1, "loss2", loss2, (1 - self.p_drop) / (self.tau * self.N))
        return loss1 + loss2 + loss3    
    
    def log_prior_loss(self):
        weights = torch.cat([parameter.flatten() if parameter.requires_grad else torch.tensor([]) for parameter in self.parameters()])
        prior_loss = torch.pow(weights, 2).sum() * self.weight_decay # (1 - self.p_drop) / (2 * self.N)
        return prior_loss / self.win_size
    
    def adversarial_sample(self, x, loss):
        if x.requires_grad == False:
            return None
        loss.backward(retain_graph=True)
        x_adv = x + 2 * torch.sign(x.grad)
        return x_adv

    def init_weight(self, anchored_weight_mu = 0, anchored_weight_std = 1):
        # weight
        nn.init.kaiming_normal_(self.conv1.weight)
        nn.init.kaiming_normal_(self.conv2.weight)
        nn.init.kaiming_normal_(self.conv3.weight)
        nn.init.kaiming_normal_(self.conv4.weight)
        nn.init.kaiming_normal_(self.conv5.weight)
        nn.init.kaiming_normal_(self.conv6.weight)
        nn.init.kaiming_normal_(self.fc1[0].weight)
        nn.init.kaiming_normal_(self.fc2[0].weight)

        # bias
        # nn.init.kaiming_normal_(self.conv1.bias)
        # nn.init.kaiming_normal_(self.conv2.bias)
        # nn.init.kaiming_normal_(self.conv3.bias)
        # nn.init.kaiming_normal_(self.conv4.bias)
        # nn.init.kaiming_normal_(self.conv5.bias)
        # nn.init.kaiming_normal_(self.conv6.bias)
        # nn.init.kaiming_normal_(self.fc1[0].bias)
        # nn.init.kaiming_normal_(self.fc2[0].bias)

    def init_hyperparameters(self, N = 3000):
        self.tau = 0.04
        self.prior_length_scale2_conv = 1e-1
        self.prior_length_scale2_linear = 1e-3
        self.N = 2000

    def stop_var_weight_update(self):
        for name, param in self.fc4.named_parameters():
            param.requires_grad = False
            print(f"fc4: Layer: {name}, requires_grad: {param.requires_grad}")


class MLP(torch.nn.Module):

  def __init__(self, win_size):
    super().__init__()
    self.win_size = win_size
    self.log_var_net = nn.Sequential(
            nn.Linear(in_features=self.win_size, out_features= 8 * self.win_size),
             nn.Tanh(),
            nn.Linear(in_features= 8 * self.win_size, out_features= 8 * self.win_size),
             nn.Tanh(),
            nn.Linear(in_features= 8 * self.win_size, out_features=self.win_size),
            nn.ReLU()
        )
  def forward(self, x):
    
    return self.log_var_net(x)
  

  
