"""
@File    :   trainer.py
@Time    :   2024/06/30 00:24:23
@Author  :   RayLam
@Contact :   1027196450@qq.com
"""

import sys

if "." not in sys.path:
    sys.path.append(".")

import time
import os
import os.path as osp

from tqdm import tqdm
from PIL import Image
import numpy as np
import torch
from torch.nn import functional as F
import torchvision
from torchvision import datasets, transforms
from torchvision.transforms import (
    Compose,
    Resize,
    ToTensor,
    Normalize,
    InterpolationMode,
)
import thop
from torchsummary import summary

from rlmc.fileop.yaml_op import Yaml
from rlmc.utils.logger import Logger
from rlmc.trainer.loss import loss_method
from rlmc.trainer.opt import optimizer
from rlmc.data.utils import onehot2mask
from rlmc.trainer.met import metrics
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score



__all__ = ["Trainer","Predict"]


logger = Logger(__name__, level=Logger.INFO)



class BaseTrainer:
    def __init__(self):
        ...
    
    def model_summary(self):
        print("model summary:")
        summary(
            self.model,
            (
                self.args.model.in_channels,
                self.args.dataset.input_size.height,
                self.args.dataset.input_size.width,
            ),
        )
        input = torch.randn(
            1,
            self.args.model.in_channels,
                self.args.dataset.input_size.height,
                self.args.dataset.input_size.width,
        ).to(self.device)
        macs, params = thop.profile(self.model, inputs=(input,))
        print("FLOPs:", macs)  # FLOPs模型复杂度
        print("params:", params)  # params参数量
        macs, params = thop.clever_format([macs, params], "%.3f")
        print("FLOPs:", macs)
        print("params:", params)

class Trainer(BaseTrainer):
    def __init__(self, configs, model, train_dataset, val_dataset, device):
        self.args = configs
        self.model = model.to(device)
        timestamp = time.time()
        self.log_time = time.strftime("%Y_%m_%d_%H_%M", time.localtime(timestamp))
        self.ckpt_dir = osp.join(self.args.train.save_dir, self.log_time)
        os.makedirs(self.ckpt_dir, exist_ok=True)
        Yaml().write(self.args.to_dict(), osp.join(self.ckpt_dir, "log_configs.yaml"))
        self.train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.args.train.dataloader.batch_size,
            shuffle=self.args.train.dataloader.shuffle,
            num_workers=self.args.train.dataloader.num_workers,
            pin_memory=self.args.train.dataloader.pin_memory,
            drop_last=self.args.train.dataloader.drop_last,
        )

        self.val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=self.args.eval.dataloader.batch_size,
            shuffle=self.args.eval.dataloader.shuffle,
            num_workers=self.args.eval.dataloader.num_workers,
            pin_memory=self.args.eval.dataloader.pin_memory,
            drop_last=self.args.eval.dataloader.drop_last,
        )

        self.device = device
        self.criterion = loss_method[self.args.train.loss]
        self.optimizer = optimizer[self.args.train.optimizer.name](
            self.model.parameters(),
            lr=self.args.train.optimizer.lr,
            weight_decay=self.args.train.optimizer.weight_decay,
        )
        
        self.metrics = metrics[self.args.train.metrics](self.args.dataset.num_classes)

        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.optimizer,
            self.args.train.optimizer.milestones,
            gamma=self.args.train.optimizer.gamma,
            last_epoch=-1,
        )
        if self.args.train.show_model_summary:
            self.model_summary()

    def train(self):
        for epoch in range(self.args.train.max_epochs):
            self.model.train()
            for batch_idx, (data, target) in enumerate(self.train_loader):
                data, target = data.to(self.device), target.to(self.device)
               
                self.optimizer.zero_grad()
                output = self.model(data)
                
                # metric_pred=output.clone().detach().cpu()
                # metric_tgt=target.clone().detach().view(-1).cpu().numpy().astype(np.int8)
                # metric_pred=torch.unsqueeze(torch.argmax(metric_pred,dim=1),dim=1)
                # tgt_shape=target.shape
                # mask=torch.zeros(tgt_shape[0],tgt_shape[1],tgt_shape[2],tgt_shape[3])
                # mask=mask.scatter(1, metric_pred,1).view(-1).numpy().astype(np.int8)
                
                ########### metrics ############
                metric_pred=output.clone().detach()
                metric_pred=torch.argmax(metric_pred,dim=1).view(-1).cpu().numpy()
                metric_tgt=target.clone().detach()
                metric_tgt=torch.argmax(metric_tgt,dim=1).view(-1).cpu().numpy()
                self.metrics.addBatch(metric_pred, metric_tgt)
                ########### metrics ############

                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()

                if batch_idx % self.args.train.log_interval == 0:
                    # accuracy=accuracy_score(metric_tgt,mask)
                    # precision=precision_score(metric_tgt,mask)
                    # recall=recall_score(metric_tgt,mask)
                    # f1=f1_score(metric_tgt,mask)
                    # logger.info(
                    #     f"Train Epoch: {epoch} [{batch_idx}/{len(self.train_loader)} "
                    #     + f"({100. * batch_idx / len(self.train_loader):.2f}%)]\tLoss: {loss.item():.6f}\tAcc: {accuracy}\tPrecision: {precision}\tRecall: {recall}\tf1: {f1}"
                    # )
                    
                    ########### metrics ############
                    acc = self.metrics.pixelAccuracy()
                    macc =self.metrics.meanPixelAccuracy()
                    mIoU = self.metrics.meanIntersectionOverUnion()
                    fwIoU = self.metrics.Frequency_Weighted_Intersection_over_Union()
                    self.metrics.reset()
                    ########### metrics ############
                    
                    logger.info(
                        f"Train Epoch: {epoch} [{batch_idx}/{len(self.train_loader)}] "
                        + ' iter_progress={:.2f}%  loss={:.6f}, acc={:.2f}%, macc={:.2f}%, mIoU={:.2f}%, fwIoU={:.2f}%'.format(100 * batch_idx / len(self.train_loader) ,loss.item(), acc*100, macc*100, mIoU*100, fwIoU*100)
                    )
            if self.args.train.is_eval and epoch !=0:
                if (epoch % self.args.train.eval_interval == 0) or epoch==self.args.train.max_epochs-1:
                    self.evaluate(epoch, is_training=True)

    def evaluate(self, epoch=10000000, is_training=False):
        if not is_training:
            state_dict=torch.load(self.args.eval.restore_model_path)
            self.model.load_state_dict(state_dict, strict=True)
        self.model.eval()
        with torch.no_grad():
            for val_batch_idx, (val_data, val_target) in enumerate(self.val_loader):
                val_data, val_target = val_data.to(self.device), val_target.to(
                    self.device
                )
                val_output = self.model(val_data)
                
                ########### metrics ############
                metric_pred=val_output.clone().detach()
                metric_pred=torch.argmax(metric_pred,dim=1).view(-1).cpu().numpy()
                metric_tgt=val_target.clone().detach()
                metric_tgt=torch.argmax(metric_tgt,dim=1).view(-1).cpu().numpy()
                self.metrics.addBatch(metric_pred, metric_tgt)
                ########### metrics ############
                
                val_loss = self.criterion(val_output, val_target)
        
        if is_training:
            ########### metrics ############
            acc = self.metrics.pixelAccuracy()
            macc =self.metrics.meanPixelAccuracy()
            mIoU = self.metrics.meanIntersectionOverUnion()
            fwIoU = self.metrics.Frequency_Weighted_Intersection_over_Union()
            self.metrics.reset()
            ########### metrics ############
            logger.info(
                f"Val Epoch: {epoch} [{val_batch_idx}/{len(self.val_loader)}] "
                + ' iter_progress={:.2f}%  loss={:.6f}, acc={:.2f}%, macc={:.2f}%, mIoU={:.2f}%, fwIoU={:.2f}%'.format(100 * val_batch_idx / len(self.val_loader) ,val_loss.item(), acc*100, macc*100, mIoU*100, fwIoU*100)
            )

            
            keys=[]
            state_dict=self.model.state_dict()
            for k,v in state_dict.items():
                if 'total_' in k:    #将torchsummary或thop生成的含total_的Unexpected key过滤掉，避免eval或predict设置为strict=True时读取模型state_dict报错
                    continue
                keys.append(k)

            new_dict = {k:state_dict[k] for k in keys}

            torch.save(
                new_dict,
                osp.join(self.ckpt_dir, f"model_{epoch:04d}.pth"),
            )
        else:
            logger.info(f"Val Loss: {val_loss.item():.6f}")
            

class Predict(BaseTrainer):
    def __init__(self, configs, model, device):
        self.args = configs
        self.model = model
        self.device = device
        self.predict_transform = Compose(
            [
                Resize(
                    (self.args.dataset.input_size.height, self.args.dataset.input_size.width),
                    interpolation=InterpolationMode.BICUBIC,
                ),
                ToTensor(),
                Normalize(
                    self.args.dataset.transformNormalize.mean,
                    self.args.dataset.transformNormalize.std,
                ),
            ]
        )
        os.makedirs(self.args.predict.save_dir, exist_ok=True)
        if self.args.predict.model_summary:
            self.model_summary()
        state_dict=torch.load(self.args.predict.restore_model_path)
        self.model.load_state_dict(state_dict, strict=True)  
        self.model.to(self.device)
        self.model.eval()
        self.palette=np.array([[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128],
              [128, 0, 128], [0, 128, 128], [128, 128, 128], [64, 0, 0],
              [192, 0, 0], [64, 128, 0], [192, 128, 0], [64, 0, 128],
              [192, 0, 128], [64, 128, 128], [192, 128, 128], [0, 64, 0],
              [128, 64, 0], [0, 192, 0], [128, 192, 0], [0, 64, 128]]).astype(np.uint8)

        
    def predict(self, img_path, is_save=False, is_show=True):
        filename=osp.splitext(osp.basename(img_path))[0]
        with torch.no_grad():
            img = Image.open(img_path)
            width,height=img.size
            img = self.predict_transform(img).unsqueeze(0).to(self.device)
            output = self.model(img)
            
            output=onehot2mask(output,axis=1)
            output=output.to('cpu')[0].numpy().astype(np.uint8)

            output=Image.fromarray(output)
            output=output.resize((width,height),Image.NEAREST)
            
            print('output',output)
        #    output[output>class_num]=0
            output.putpalette(self.palette)
            if is_save:
                output.save(osp.join(self.args.predict.save_dir,filename + '.png'))

            if is_show:
                output.show()


