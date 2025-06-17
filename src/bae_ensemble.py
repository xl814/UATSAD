import torch
import numpy as np 
import torch.nn as nn

class BAE_Ensemble():
    def __init__(self, num_samples, Module, config, randomseed=None):

        super().__init__()
        self.num_samples = num_samples
        self.lr = config.lr

        # init ensembls
        if randomseed == None:
            self.aes = [Module(config) for _ in range(num_samples)]
        self.optimizers = [torch.optim.Adam(model.mean_net.parameters(), self.lr) for model in self.aes]

    def tunning_var(self, lr):
        self.optimizers = [torch.optim.Adam(model.parameters(), lr) for model in self.aes]

    def set_rnll_hyper(self, lambda1, lambda2):
        for model in self.aes:
            model.set_rnll_hyper(lambda1, lambda2)

    def set_loss_type(self, loss_type):
        for model in self.aes:
            model.set_loss_type(loss_type)

    def set_punishment(self, punishment):
        for model in self.aes:
            model.set_punishment(punishment)

    def fit(self, x, batch=-1):
        stacked_criterion = torch.stack(
            [
                model.criterion(x, batch) for model in self.aes
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
        for model in self.aes:
            model.train()

    def eval(self):
        for model in self.aes:
            model.eval()

    def predict_mean(self, x):
        stacked_pred = torch.stack(
            [
                model.predict(x)[0] for model in self.aes
            ]
        )
        # print(stacked_pred.mean(dim=0).shape)
        return stacked_pred.mean(dim=0)

    def predict_var(self, x):
        predictions = []
        var = []
        for model in self.aes:
            output = model.predict(x)
            predictions.append(output[0].detach().numpy())
            var.append(output[1].detach().numpy())

        predictions = torch.tensor(np.array(predictions))
        var = torch.tensor(np.array(var))
        # print(predictions.shape) #(10, batch_size, 1, win_size)
        return predictions.mean(dim=0), var
    
    def predict(self, x):
        '''
            NOTE: This predict only returns `cpu` value
        '''
        predictions = []
        var = []
        for model in self.aes:
            output = model.predict(x)
            predictions.append(output[0].cpu().detach().numpy())
            var.append(output[1].cpu().detach().numpy())

        predictions = torch.tensor(np.array(predictions))
        var = torch.tensor(np.array(var))
        # print(predictions.shape) #(10, batch_size, 1, win_size)
        return predictions, var
    
    def predict_mean_var(self, x):
        stacked_pred = torch.stack(
            [
                model.predict(x)[0] for model in self.aes
            ]
        )
        stacked_var = torch.stack(
            [
                model.predict(x)[1] for model in self.aes
            ]
        )
        return stacked_pred.mean(dim=0), stacked_var.mean(dim=0)
    
    def predict_stack_values(self, x):
        stacked_pred = torch.stack(
            [
                model.predict(x)[0] for model in self.aes
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
        for model in self.aes:
            model.to(device)

    def save(self, path):
        for i, model in enumerate(self.aes):
            torch.save(model.state_dict(), path + f"ensemble_{i}.pth",)

    def load(self, path):
        for i, model in enumerate(self.aes):
            model.load_state_dict(torch.load(path + f"ensemble_{i}.pth", weights_only=True))