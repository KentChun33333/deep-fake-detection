from pytorch_lightning import Trainer
from argparse import Namespace

import pytorch_lightning as pl
from torch import Tensor
from torch import nn
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset

from model_classify import PairImgClasssifer_DeepFace, PairImgClasssifer_DeepFace_Multi
from ka_dataset import Ka2020DeepFack, Ka2020DeepFackMulti

class PairClasssifer_DeepFaceMultiSys(pl.LightningModule):

    def __init__(self, hparams):
        super().__init__()
        # train, test, train_labels, 
        self.hparams = hparams
        # change to 2 node 
        self.model = PairImgClasssifer_DeepFace_Multi()
        
        self.frame_sampler_para = hparams.frame_sampler_para
        self.hparams.frame_sampler_para = {'mode' : 'multi', 'N':7  }    

        self.batch_size = hparams.batch_size
        self.mctnn_size = hparams.mctnn_size
        self.outimg_size = hparams.outimg_size
        self.learning_rate = hparams.learning_rate

        self.shuffle = True
        self.criterion = nn.CrossEntropyLoss()
        
    def build_dataset(self, train_df, valid_df, test_df, mtcnn):

        self.train_dataset = Ka2020DeepFackMulti(train_df,  
            self.frame_sampler_para, face_defector_mctnn=mtcnn, 
            mctnn_size=self.mctnn_size, 
            outimg_size=self.outimg_size)

        self.vaild_dataset = Ka2020DeepFackMulti(valid_df,  
            self.frame_sampler_para, face_defector_mctnn=mtcnn, 
            mctnn_size=self.mctnn_size, 
            outimg_size=self.outimg_size)

        self.test_dataset = Ka2020DeepFackMulti(test_df,  
            self.frame_sampler_para, face_defector_mctnn=mtcnn, 
            mctnn_size=self.mctnn_size, 
            outimg_size=self.outimg_size)

        
    def sys_forward(self, i):
        pre_imgs = i['frames'][:, 0, :, :, :, :].permute(0, 1, 4, 2, 3).float()
        post_imgs = i['frames'][:, 1, :, :, :, :].permute(0, 1, 4, 2, 3).float()

        y_hat = self.model(  pre_imgs, post_imgs)
        #_, y_hat = torch.max(y_hat, dim=1)
        return y_hat

    def get_loss(self, y_hat, y):
        loss = self.criterion(y_hat, y)
        return loss

    def training_step(self, batch, batch_idx):
        # REQUIRED
        # res = {'frames': [], 'frame_ids': [], 'land_marks':[] }
        # res['frames'].shape ; torch.Size([4, 2, 256, 256, 3])

        y = batch['target'].long()
        y_hat = self.sys_forward(batch)
        loss = self.get_loss(y_hat, y)
        tensorboard_logs = {'train_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx):

        # NHWC or NCHW
        y = batch['target'].long()
        y_hat = self.sys_forward(batch)
        loss = self.get_loss(y_hat, y)
        
        return {'val_loss': loss, }

    def test_step(self, batch, batch_idx):
        y = batch['target'].long()
        
        y_hat = self.sys_forward(batch)

        loss = self.get_loss(y_hat, y)
        
        return {'val_loss': loss, }

    def validation_end(self, outputs):
        # OPTIONAL
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss}
        return { 'val_loss': avg_loss, 'progress_bar':{'val_loss': avg_loss}}
        
    def configure_optimizers(self):
        # REQUIRED
        # can return multiple optimizers and learning_rate schedulers
        # (LBFGS it is automatically supported, no need for closure function)
        return torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

    @pl.data_loader
    def train_dataloader(self):
        # REQUIRED
        return DataLoader(self.train_dataset,
                      self.batch_size,
                      self.shuffle,
                      pin_memory=True, 
                      collate_fn=self.train_dataset.my_collate) 

    @pl.data_loader
    def val_dataloader(self):
        # OPTIONAL
        return DataLoader(self.vaild_dataset,
                      self.batch_size,
                      self.shuffle,
                      pin_memory=True, 
                      collate_fn=self.vaild_dataset.my_collate) 
    
    @pl.data_loader
    def test_dataloader(self):
        # OPTIONAL
        return DataLoader(self.test_dataset,
                      self.batch_size,
                      self.shuffle,
                      pin_memory=True, 
                      collate_fn=self.test_dataset.my_collate) 

def dfs_freeze(model):
    for name, child in model.named_children():
        for param in child.parameters():
            param.requires_grad = False
        dfs_freeze(child)

def dfs_unfreeze(model):
    for name, child in model.named_children():
        for param in child.parameters():
            param.requires_grad = True
        dfs_freeze(child)
