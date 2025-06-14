import torch
import numpy as np 
import torch.nn as nn
import utils

class BAE_Ensemble():
    def __init__(self, num_samples, VAEModule, config, randomseed=None):

        super().__init__()
        self.num_samples = num_samples
        self.lr = config.lr
        # self.path = "/home/new_lab/test/ensemble_bae/saved_taxi_5_5/"
        self.path = "/home/new_lab/test/ensemble_bae/saved_5_4_SMAP/"
        # self.path = "/home/new_lab/test/ensemble_bae/saved_swat_5_6/"

        # init VAE modules
        if randomseed == None:
            self.vaes = [VAEModule(config) for _ in range(num_samples)]
        self.optimizers = [torch.optim.Adam(model.mean_net.parameters(), self.lr) for model in self.vaes]

    def set_optimizers_lr(self, lr):
        self.optimizers = [torch.optim.Adam(model.log_var_net.parameters(), lr) for model in self.vaes]

    def reset_optimizers_for_logvar(self):
        self.optimizers = [torch.optim.SGD([{'params': self.log_var}], self.lr * 0.1) for model in self.vaes]

    def tunning_var(self, lr):
        self.optimizers = [torch.optim.Adam(model.parameters(), lr) for model in self.vaes]
        # for i, opti in enumerate(self.optimizers):
        #     opti.add_param_group({'params': self.vaes[i].mean_net.parameters(), 'lr': 1e-4})
        # for optimizer in self.optimizers:
        #     # print("hhh", len(optimizer.param_groups))
        #     for param_group in optimizer.param_groups:
        #         param_group['lr'] = lr
    def set_rnll_hyper(self, lambda1, lambda2):
        for model in self.vaes:
            model.set_rnll_hyper(lambda1, lambda2)

    def set_loss_type(self, loss_type):
        for model in self.vaes:
            model.set_loss_type(loss_type)

    def set_punishment(self, punishment):
        for model in self.vaes:
            model.set_punishment(punishment)

    def fit(self, x, batch=-1):
        stacked_criterion = torch.stack(
            [
                model.criterion(x, batch) for model in self.vaes
            ]
        )
        return stacked_criterion.mean()
    
    def step_optimizers(self):
        for optimizer in self.optimizers:
            optimizer.step()
    
    def zero_optimizers(self):
        for optimizer in self.optimizers:
            optimizer.zero_grad() 

    def train(self):
        for model in self.vaes:
            model.train()

    def eval(self):
        for model in self.vaes:
            model.eval()

    def predict_mean(self, x):
        stacked_pred = torch.stack(
            [
                model.predict(x)[0] for model in self.vaes
            ]
        )
        # print(stacked_pred.mean(dim=0).shape)
        return stacked_pred.mean(dim=0)

    def predict_logvar(self, x):
        predictions = []
        log_var = []
        for model in self.vaes:
            output = model.predict(x)
            predictions.append(output[0].detach().numpy())
            log_var.append(output[1].detach().numpy())

        predictions = torch.tensor(np.array(predictions))
        log_var = torch.tensor(np.array(log_var))
        # print(predictions.shape) #(10, batch_size, 1, win_size)
        return predictions.mean(dim=0), log_var
    
    def predict(self, x):
        '''
            NOTE: This predict only returns `cpu` value
        '''
        predictions = []
        log_var = []
        for model in self.vaes:
            output = model.predict(x)
            predictions.append(output[0].cpu().detach().numpy())
            log_var.append(output[1].cpu().detach().numpy())

        predictions = torch.tensor(np.array(predictions))
        log_var = torch.tensor(np.array(log_var))
        # print(predictions.shape) #(10, batch_size, 1, win_size)
        return predictions, log_var
    
    def predict_mean_logvar(self, x):
        stacked_pred = torch.stack(
            [
                model.predict(x)[0] for model in self.vaes
            ]
        )
        stacked_logvar = torch.stack(
            [
                model.predict(x)[1] for model in self.vaes
            ]
        )
        return stacked_pred.mean(dim=0), stacked_logvar.mean(dim=0)
    
    def predict_stack_values(self, x):
        stacked_pred = torch.stack(
            [
                model.predict(x)[0] for model in self.vaes
            ]
        )
        return stacked_pred
    
    def adjust_lr(self, epoch):
        lr_adjust = {
            10: 1e-4,
            15: 5e-4, 20: 1e-5
        }
        if epoch in lr_adjust.keys():
            for optimizer in self.optimizers:
                for param_group in optimizer.param_groups:
                    lr = lr_adjust[epoch]
                    param_group['lr'] = lr
        # print('Updating learning rate to {}'.format(lr))


    def toDevice(self, device):
        for model in self.vaes:
            model.to(device)

    def save(self):
        for i, model in enumerate(self.vaes):
            torch.save(model.state_dict(), self.path + f"ensemble_{i}.pth",)

    def load(self):
        for i, model in enumerate(self.vaes):
            model.load_state_dict(torch.load(self.path + f"ensemble_{i}.pth", weights_only=True))