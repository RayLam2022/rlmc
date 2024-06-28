import sys

if "." not in sys.path:
    sys.path.append(".")

import os
import os.path as osp

from tqdm import tqdm
import numpy as np
import torch
import torchvision

from rlmc.utils.logger import Logger


from rlmc.model.cv import *


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger = Logger(__name__, level=Logger.INFO)

class Trainer:
    def __init__(self, epochs, save_checkpoint_interval, eval_interval, dataloader, model, optimizer, scheduler, criterion): 
        self.epochs=epochs
        self.save_checkpoint_interval=save_checkpoint_interval
        self.eval_interval=eval_interval
        self.device=device
        self.dataloader=dataloader
        self.model=model
        self.optimizer=optimizer
        self.criterion=criterion
        self.scheduler=scheduler

    def train(self, epochs, save_checkpoint_interval, eval_interval):
        for epoch in range(epochs):
            self.model.train()
            for batch_idx, (data, target) in enumerate(self.dataloader):    
                data, target = data.to(self.device), target.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output,target)
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()
                
                if batch_idx % 10 == 0:




if __name__ == "__main__":

    model.to(device)

    do_eval=True
    log_interval=10
    criterion = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True) #smp.losses.DiceLoss(mode='multilabel')#nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.01)

    num_epochs = 200
    for epoch in range(num_epochs):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)

            #output = (output == output.max(dim=1, keepdim=True)[0])
            # print('tgt',target.shape,target.device)
            # print('out',output.shape,output.device)
            # print(np.unique(target.cpu().numpy()))
            loss = criterion(output,target)
            loss.backward()
            optimizer.step()
            scheduler.step()

            if batch_idx % log_interval == 0:
                print(f"Train Epoch: {epoch} [{batch_idx}/{len(train_loader)} "
                    f"({100. * batch_idx / len(train_loader):.2f}%)]\tLoss: {loss.item():.6f}")
                
        if epoch % 20==0 and epoch!=0:
            if do_eval:
                model.eval()
                with torch.no_grad():
                    for val_batch_idx, (val_data, val_target) in enumerate(val_loader):
                        val_data, val_target = val_data.to(device), val_target.to(device)
                        val_output = model(val_data)
                        val_loss = criterion(val_output,val_target)
                print(f"Val Epoch: {epoch} [{val_batch_idx}/{len(val_loader)} "
                        f"({100. * val_batch_idx / len(val_loader):.2f}%)]\tLoss: {val_loss.item():.6f}")
            torch.save(model.state_dict(), osp.join(save_ckpt_path,f'model_{epoch:04d}.pth'))
