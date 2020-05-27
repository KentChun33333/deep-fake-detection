
from pytorch_lightning import Trainer
from argparse import Namespace

import pytorch_lightning as pl
from torch import Tensor
from torch.autograd import Variable

from torch import nn
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset

from ka_dataset_noise import Ka2020DeepFackNoise


from ffdnet import FFDNet

class Meso4(nn.Module):
    """
    Pytorch Implemention of Meso4
    Autor: Honggu Liu
    Date: July 4, 2019
    """
    def __init__(self, num_classes=2):
        super(Meso4, self).__init__()
        self.num_classes = num_classes
        self.conv1 = nn.Conv2d(3, 8, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(8)
        self.relu = nn.ReLU(inplace=True)
        self.leakyrelu = nn.LeakyReLU(0.1)

        self.conv2 = nn.Conv2d(8, 8, 5, padding=2, bias=False)
        self.bn2 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(8, 16, 5, padding=2, bias=False)
        self.conv4 = nn.Conv2d(16, 16, 5, padding=2, bias=False)
        self.maxpooling1 = nn.MaxPool2d(kernel_size=(2, 2))
        self.maxpooling2 = nn.MaxPool2d(kernel_size=(4, 4))
        #flatten: x = x.view(x.size(0), -1)
        self.dropout = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(3136, 16)
        self.fc2 = nn.Linear(16, num_classes)

    def forward(self, x):
        x = self.conv1(x) #(8, 256, 256)
        x = self.relu(x)
        x = self.bn1(x)
        x = self.maxpooling1(x) #(8, 128, 128)

        x = self.conv2(x) #(8, 128, 128)
        x = self.relu(x)
        x = self.bn1(x)
        x = self.maxpooling1(x) #(8, 64, 64)

        x = self.conv3(x) #(16, 64, 64)
        x = self.relu(x)
        x = self.bn2(x)
        x = self.maxpooling1(x) #(16, 32, 32)

        x = self.conv4(x) #(16, 32, 32)
        x = self.relu(x)
        x = self.bn2(x)
        x = self.maxpooling2(x) #(16, 8, 8)

        x = x.view(x.size(0), -1) #(Batch, 16*8*8)
        x = self.dropout(x)
        x = self.fc1(x) #(Batch, 16)
        x = self.leakyrelu(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x

class FFDwrap(nn.Module):
    def __init__(self):
        super().__init__()
        self.module  = FFDNet()

    def forward(self, x, y):
        return self.module(x, y)

class FFD_Meso_Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.ffd_model = FFDwrap()
        self.ffd_model.load_state_dict(torch.load('net_rgb.pth'))
        dfs_freeze(self.ffd_model)

        self.meso_model = Meso4()

    def forward(self, imgs, noise_sigma):
        noise_img = self.ffd_model(imgs, noise_sigma) 
        # in ffd model there is a implementation restriction... to have it only accept gpu 0 
        y = self.meso_model(noise_img.to('cuda:1'))
        #y, ind = torch.max(y, dim=1)
        return y

class DeepFaceNoise_SYS(pl.LightningModule):

    def __init__(self, hparams):
        super().__init__()
        # train, test, train_labels, 
        self.hparams = hparams
        # change to 2 node 
        self.model = FFD_Meso_Net()
        
        hparams.frame_sampler_para = {'mode' : 'one' }    


        self.frame_sampler_para = hparams.frame_sampler_para
        self.batch_size = hparams.batch_size
        self.outimg_size = hparams.outimg_size
        self.learning_rate = hparams.learning_rate

        self.shuffle = True
        self.criterion = nn.CrossEntropyLoss()
        
    def build_dataset(self, train_df, valid_df, test_df):

        self.train_dataset = Ka2020DeepFackNoise(train_df,  
            self.frame_sampler_para,  
            outimg_size=self.outimg_size)

        self.vaild_dataset = Ka2020DeepFackNoise(valid_df,  
            self.frame_sampler_para,  
            outimg_size=self.outimg_size,  )

        self.test_dataset = Ka2020DeepFackNoise(test_df,  
            self.frame_sampler_para,  
            outimg_size=self.outimg_size,  )

        
    def sys_forward(self, i):
        x = i['frames']#.to('cuda:1')
        y = i['noise_sigma']#.to('cuda:1')
        y_hat = self.model(x, y)
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

class FFDnetSmoother():
    def __init__(self, model_path, add_noise=False, 
        noise_sigma=40, with_gpu=False, in_ch=3, ):
        '''
        Args : 
          - add_noise : default fasle 
          - noise_sigma : 40 ~100 could be good 
          - with_gpu : True or False 
          - inch : 1 for gray, 3 for RGB
        '''

        # super().__init__()

        self.add_noise = add_noise
        self.noise_sigma = noise_sigma
        self.with_gpu = with_gpu
        self.noise_sigma = noise_sigma/255.
        self.in_ch = in_ch

        if in_ch==3:
            model_fn = 'models/net_rgb.pth'
        else:
            model_fn = 'models/net_gray.pth'

        self.model_fn = model_path

        self._load_model()

    def _load_model(self):
        print('Loading model ...\n')
        net = FFDNet(num_input_channels=self.in_ch, test_mode=True)    

        # Load saved weights
        if self.with_gpu :
            state_dict = torch.load(self.model_fn)
            # device_ids = [0]
            # model = nn.DataParallel(net, device_ids=device_ids)
            dtype = torch.cuda.FloatTensor
        else:
            state_dict = torch.load(self.model_fn, map_location='cpu')
            # CPU mode: remove the DataParallel wrapper
            state_dict = remove_dataparallel_wrapper(state_dict)
            model = net
            dtype = torch.FloatTensor    

        model.load_state_dict(state_dict)    

        # Sets the model in evaluation mode (e.g. it removes BN)
        model.eval()
        self.model  = model
        self.dtype  = dtype

    def extra_noitse(self, img):
        if len(img.shape)==2: # single gray image 
            img = np.expand_dims(img, 0)
        if len(img.shape) ==3 : # single image
            img = np.expand_dims(img, 0)
        
        expanded_h = False
        expanded_w = False
        sh_im = img.shape
        if sh_im[2]%2 == 1:
            expanded_h = True
            img = np.concatenate((img, 
                    img[:, :, -1, :][:, :, np.newaxis, :]), axis=2)    

        if sh_im[3]%2 == 1:
            expanded_w = True
            img = np.concatenate((img, 
                    img[:, :, :, -1][:, :, :, np.newaxis]), axis=3)    

        img = np.float32(img/255.)
        img = torch.Tensor(img)

        if self.add_noise:
            noise = torch.FloatTensor(img.size()).normal_(mean=0, std=noise_sigma)
            imnoisy = img + noise
        else:
            imnoisy = img.clone()

        # infrence
        with torch.no_grad():
            # with torch.no_grad(): # PyTorch v0.4.0
            img     = Variable(img.type(self.dtype))
            imnoisy = Variable(imnoisy.type(self.dtype))
            nsigma  = Variable(torch.FloatTensor(
                              [self.noise_sigma]).type(self.dtype))        

            # Measure runtime
            start_t = time.time()        

            # Estimate noise and subtract it to the input image
            im_noise_estim = self.model(imnoisy, nsigma)
        return im_noise_estim
