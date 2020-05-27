
from argparse import Namespace

import pytorch_lightning as pl

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset


from collections import defaultdict
from torch import nn 

from tqdm import tqdm
from copy import deepcopy
import torch, os
from torch.nn import functional as F

from dataset import Ka2020DeepFackSeq
from model import CNN_LSTM, CNN_Only, CNN_LSTM_Simple, CNN_LSTM_3D, FlattenDNN

def undersample_balance(df, col):
    res= []
    min_value = df[col].value_counts().min()
    for v in list(set(df[col])):
        temp_df = df[df[col]==v]
        res.append(temp_df.sample(n=min_value))
    return pd.concat(res, ignore_index=True)

def oversample_balance(df, col):
    res= []
    max_value = df[col].value_counts().max()
    for v in list(set(df[col])):
        temp_df = df[df[col]==v]
        res.append(temp_df.sample(n=max_value, replace=True))
    return pd.concat(res, ignore_index=True)

def select_by_original_(df):
    cc = df.groupby('original').apply(lambda x: x.sample(n=1))
    cc = [i[1] for i in cc.index]
    ddff = df.loc[cc]
    ddff.index = range(len(ddff))
    return ddff

class DeepFace_Seq_Sys(pl.LightningModule):
    # limit 
    # 
    def __init__(self, hparams):
        super().__init__()
        # train, test, train_labels, 
        self.hparams = hparams
        
        self.regist_model = {
            "CNN_LSTM"  : CNN_LSTM, 
            "CNN_Only": CNN_Only,
            "CNN_LSTM_Simple":CNN_LSTM_Simple, 
            "CNN_LSTM_3D":CNN_LSTM_3D,
            "FlattenDNN":FlattenDNN,
        }
        
        self.regist_dataset = {
            "Ka2020DeepFackSeq":Ka2020DeepFackSeq, 
            # "Ka2020DeeFackFrameSeq": Ka2020DeeFackFrameSeq, 
        }
        # change to 2 node 
        self.model = self.regist_model[hparams.model_name](hparams = hparams)
        
        self.batch_size = hparams.batch_size
        self.learning_rate = hparams.learning_rate

        self.shuffle = True

        if self.hparams.output_size==1:
            #BCELossWithLogits 
            self.criterion = nn.BCEWithLogitsLoss()
            self.criterion_ohml = nn.BCEWithLogitsLoss(reduction ='none')
        else:
            self.criterion = nn.CrossEntropyLoss()

        self.validation_result = defaultdict(list)
        self.last_training_avg_loss = 0 # init
        
    def select_true_original(self, real_df):
        select_obj = { 28 :[ 2, 3, 7, 33, 80, 93, 113, 151, 152, 155, 210, 267, 278, 323, 331, 357, 379], 




                       1 : [ 2, 23, 46, 99 ], 
                       2 : [ 0, 2, 11, 85, 92, 93, 104, 189, 202 ], 
                       3 : [ 0, 1, 3, 71, 74, 91, 111, 119, 163, 165, 168, 174, 203, 218 ], 
                       4 : [ 0, 1, 11, 72, 82, 98, 122, 128, 131, 132, 135, 167 ], 
                       5 : [  0, 6, 25, 90, 112, 128, 160, 190, 242, 246, 258, 290, 337], 
                       6 : [  2, 10, 15, 16, 145, 163, 170, 218, 223, 248, 292, 293, 295, 296, 303, 349, 350, 351, 352,386, 398, 359, 409, 414], 
                       7 : [  0, 1, 5, 16, 38, 74, 107, 162, 172, 195, 223, 225, 227, 232,257, 288, 289, 297, 319, 325], 
                       8 : [  0, 1, 2, 3, 45, 65,66, 67, 118, 122, 130, 165, 166, 167, 169, 170, 177, 190, 191, 242, 248], 
                       9 : [  1, 2, 4, 5, 52, 67, 110, 116, 117, 119, 127, 188, 194, 230, 233, 251, 280], 
                       10 : [ 0, 1, 2, 4, 5, 14, 19, 31, 45, 58, 101, 106, 144, 165, 166, 169, 170, 236, 237, 256, 276, 277, 279, 291, 307, 313, 326, 331], 
                       11 : [ 0, 1, 2, 10, 14, 19, 22, 27, 58, 59, 62, 96, 109, 111, 117, 127, 130, 133, 134, 138, 144, 157, 189, 190, 192, 223, 224, 230, 237, 259, 291, 292, 293, 300, 306, 328, 344, 345, 352], 
                       12 : [ 0, 1, 2, 3, 4, 6, 7, 10, 11, 12, 22, 24, 25, 41, 44, 57, 58, 72, 74, 142, 146, 147, 194, 196, 197, 199, 202, 203, 213, 216, 219, 220, 263, 265, 266, 283, 295, 296, 311 ], 
                       13 : [ 0, 1, 8, 13, 17, 22, 26, 34, 35, 36, 39, 48, 50, 51, 100, 101, 102, 103, 104, 106, 108, 122, 141, 159, 164, 166, 167, 168, 178, 212, 221, 226, 227, 236, 224, 254, 259, 271, 273, 279, 327, 328, 331, 363, 364, 369, 373, 374, 377, 390, 406] 
                       }
        
        keys = select_obj.keys()
        res = []
        for i in set(real_df.chunk):
            df_temp = real_df[real_df['chunk']==i]

            df_temp.index = range(len(df_temp))
            if i in keys:
                inds = select_obj[i]    
            else:
                inds = [i for i in range(len(df_temp)) if i%4==0]

            res.append(df_temp.iloc[inds])
        return pd.concat(res, ignore_index=True)


    def build_dataset(self, train_df, valid_df, test_df ):
        # for dynamically rebuild the dataset 

        self.train_df = deepcopy(train_df) # this is raw train df 

        def balance_df(df):

            real_df = df[df['target']==0]

            real_df = self.select_true_original(real_df)

            fake_df = df[df['target']==1]

            fake_df = fake_df[fake_df.original.isin(list(real_df.original))]
        
            print(len(real_df), len(fake_df))
        
            fake_df = select_by_original_(fake_df) # one original only show one times 
            return  pd.concat([real_df, fake_df], ignore_index=True)

        train_df = balance_df(train_df)
        valid_df = balance_df(valid_df)
        test_df = balance_df(valid_df)
        #print(set(train_df.original.value_counts()))

        print('-------------------------------')
        print(train_df.target.value_counts())

        self.train_dataset = self.regist_dataset[self.hparams.dataset_name](train_df, phase= 'train')
        self.vaild_dataset = self.regist_dataset[self.hparams.dataset_name](valid_df, phase='valid')
        self.test_dataset = self.regist_dataset[self.hparams.dataset_name](test_df, phase='test')

        
    def forward(self, batch):
        # chose only one face -w- shit 
        # [5, 10, 224, 224, 3]
        #print(imgs['frames'])
        seq_min = self.hparams.seq_min
        seq_max = self.hparams.seq_max

        x =  batch['frames'][:, seq_min:seq_max, :, :, : ]

        x = x.permute(0, 1, 4, 3, 2)

        y_hat = self.model( x)
        y_hat = torch.clamp(y_hat, 0.01, 0.99)
        return y_hat

    def get_loss(self, y_hat, y):
        print(y_hat, y)
        
        predict = (y_hat > 0.5).float() * 1
        
        # mix loss 
        acc = (predict==y).float().mean() * 1
        
        # only update bad
        if self.hparams.loss_func_name == '0.5 bce + 0.5 acc':
            loss = self.criterion(y_hat, y)*0.5+(1-acc)*0.5
        elif self.hparams.loss_func_name == '0.8 bce + 0.2 acc':
            loss = self.criterion(y_hat, y)*0.8+(1-acc)*0.2
        elif self.hparams.loss_func_name == 'bce':
            loss = self.criterion(y_hat, y)
        elif self.hparams.loss_func_name == 'hard_sample_weighted_bce':
            loss = self.get_loss_online_hard_mining(y_hat, y)
            
        return loss
    
    def get_loss_online_hard_mining(self, y_pre, y):
        '''
        the weighted BCE loss on not accurate prediction 
        '''
        alpha = 0.2
        loss_all = self.criterion_ohml(y_pre, y).squeeze()
        predict = (y_pre > 0.5).float() * 1
        err = (predict!=y).float().squeeze()+alpha
        err = err/(1+alpha)
        loss_all = loss_all*err
        return loss_all.mean()
        
    def training_step(self, batch, batch_idx):
        self.model.phase='train'

        if self.hparams.output_size != 1: 
            y = batch['target'].long()
        else:
            y = batch['target'].float().unsqueeze(1)
            
        y_hat = self.forward(batch)
        

        loss = self.get_loss(y_hat, y)
                
        log_dict = {'train_loss': loss}
                    
        return {'loss': loss, 'log': log_dict}
    
    def validation_step(self, batch, batch_idx):
        self.model.phase='valid'

        if self.hparams.output_size != 1: 
            y = batch['target'].long()
        else:
            y = batch['target'].float().unsqueeze(1)
        y_hat = self.forward(batch)
        
        loss = self.get_loss(y_hat, y)
        
        predict = (y_hat > 0.5).float() * 1
        
        for i in range(y_hat.size()[0]):
            self.validation_result['prediction_result'].append(y_hat[i])
            self.validation_result['file_name'].append(batch['file_name'][int(batch['batch_idx'][i].item())])
            self.validation_result['original'].append(batch['original'][i])
            self.validation_result['target'].append(batch['target'][i])
        return {'val_loss': loss, 'predict': predict.squeeze(), 'target': batch['target'].float() }
    
    def test_step(self, batch, batch_idx):
        self.model.phase='valid'

        if self.hparams.output_size != 1: 
            y = batch['target'].long()
        else:
            y = batch['target'].float().unsqueeze(1)
        y_hat = self.forward(batch)
        loss = self.get_loss(y_hat, y)

        return {'val_loss': loss }
    
    # training step end is doing within batch for parrellel training 
    def training_end(self, outputs):
        # this out is now the full size of the batch
        try:
            avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        except:
            avg_loss = outputs['loss']
        self.last_training_avg_loss = avg_loss
        # this softmax now uses the full batch size
        #loss = nce_loss(loss)
        return {'loss': avg_loss, }
    
    def validation_epoch_end(self, outputs):
#         try:
#             # this only wrok for ... 
#             if self.trainer.current_epoch <2:
#                 dfs_freeze(self.model.cnn_model)
#                 print('== dfs_freeze self.model.cnn_model  ===')
#             elif self.trainer.current_epoch ==2:
#                 dfs_unfreeze(self.model.cnn_model)
#                 print('== dfs_UNfreeze self.model.cnn_model  ===')
#         except:
#             print('no backbone cnn_model')
            
        try:
            # DP 
            avg_loss = torch.cat([x['val_loss'] for x in outputs]).mean()
            predict = torch.cat([x['predict'] for x in outputs])
            target = torch.cat([x['target'] for x in outputs])
        except:
            # Single GPU
            avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
            predict = torch.stack([x['predict'] for x in outputs])
            target = torch.stack([x['target'] for x in outputs])
        acc = (predict==target).float().mean() * 1
        
        log_dict = {'val_loss': avg_loss, 
                    'train_loss': self.last_training_avg_loss, 
                    'ep': self.current_epoch, 
                    'Acc': acc, 
                   }
        
        log_dict.update(self.hparams.__dict__)
        log_df = pd.DataFrame()
        log_df = log_df.append(log_dict, ignore_index=True)
        
        if not self.exp_save_path:
            self.exp_save_path = self.trainer.default_save_path
            
        path = os.path.join(self.exp_save_path , 'log.csv')
        
        if os.path.exists(path):
            log_df.to_csv(path , mode='a', header=False )
        else:
            log_df.to_csv(path)
        
        df = pd.DataFrame(self.validation_result)

        df.to_csv(os.path.join(self.exp_save_path , f'validation_{self.current_epoch}.csv'))
        self.validation_result = defaultdict(list)
        print('SAVE : ', os.path.join(self.exp_save_path , f'validation_{self.current_epoch}.csv'))
        
        return { 'val_loss': avg_loss, 'progress_bar':{'val_loss': avg_loss, 'Acc': acc}, }
        
    def configure_optimizers(self):
        # REQUIRED
        #opt_mizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        if self.hparams.optimizer_name=='Adam':
            opt_mizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, )
        elif self.hparams.optimizer_name=='AdamW':
            #AdamW
            opt_mizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate)
        elif self.hparams.optimizer_name=='SGD':
            opt_mizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate, )
        #opt_mizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=0.9)
        #scheduler = torch.optim.lr_scheduler.CyclicLR(opt_mizer, base_lr=self.learning_rate, max_lr=0.001, cycle_momentum=True)
        return [opt_mizer] #, [scheduler]

    def train_dataloader(self):
        # REQUIRED
        print('...')
        return DataLoader(self.train_dataset,
                      self.batch_size,
                      self.shuffle,
                      pin_memory=True, 
                      collate_fn=self.train_dataset.my_collate) 

    def val_dataloader(self):
        # OPTIONAL
        return DataLoader(self.vaild_dataset,
                      self.batch_size,
                      self.shuffle,
                      pin_memory=True, 
                      collate_fn=self.vaild_dataset.my_collate) 
    
    def test_dataloader(self):
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
    model.eval()

def dfs_unfreeze(model):
    model.train()
    for name, child in model.named_children():
        for param in child.parameters():
            param.requires_grad = True
        dfs_freeze(child)
        
def validation_of_reshape_for_back_cnn():
    
    def test_for_and_back(x, op_instance):     
        B, frame, face, C, H, W = x.size()
        x = x.reshape((B *frame *face, C, H, W))
        # 8 - output channel
        res = op_instance(x).reshape(B, frame*face*8*H*W)
        return res
    
    # init batch size with 2, 3-frames, 5-face, 3-channel, H,W 
    a = torch.randint(2, 100, (2,3, 5,3, 244, 244)).float()
    # a1 is the first sample from the batch 
    a1  = a[0].unsqueeze(0)
    
    # define a single con2d 
    ly = nn.Conv2d(3, 8, 3, padding=1, bias=False)  

    res_a = test_for_and_back(a, ly)
    res_a1 = test_for_and_back(a1, ly)
    print(res_a[0] == res_a1)