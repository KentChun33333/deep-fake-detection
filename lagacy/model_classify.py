
# Model -------------------
from torch import nn
import pickle
from skimage.transform import resize as sk_resize
from tqdm import tqdm
import sys, pickle, os, re

from skimage.io import imread, imshow, imsave
import torch
import numpy as np
import pandas as pd
from torch.nn import functional as F
from efficientnet_pytorch import EfficientNet

class PairImgClasssifer_DeepFace_Multi(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder_net = EfficientNet.from_pretrained('efficientnet-b0')
        self.binary_net = BSL_MultiNet_SEBlock_LN(2)

    def forward(self, imgs):
        # imgs :[batch, face, :, :, :].
        # model.extract_features(process_tensor_to_img(pre_img)).shape
        # [-1, 1280, 7, 7]
        _, frame_id ,face_ind, _, _, _ = imgs.size()
        res_frame = []
        for iid in range(8):
            res = []
            for ind in range(face_ind):
                pre_m = self.encoder_net.extract_features(
                    imgs[:, iid, ind, :, :, :].float())
                res.append(self.binary_net(pre_m))
            res, ind = torch.max(torch.stack(res), dim=0)
            res_frame.append(res)
        res_frame, ind = torch.max(torch.stack(res_frame), dim=0)
        return res_frame

class PairImgClasssifer_DeepFace(nn.Module):
    def __init__(self):
        super(PairImgClasssifer_DeepFace, self).__init__()
        self.encoder_net = EfficientNet.from_pretrained('efficientnet-b0')
        self.binary_net = BSL_MultiNet_SEBlock_LN(2)

    def forward(self, pre_img, post_img):
        # model.extract_features(process_tensor_to_img(pre_img)).shape
        # [-1, 1280, 7, 7]
        pre_m = self.encoder_net.extract_features(pre_img)        
        out = self.binary_net(pre_m)
        return out

class BSL_MultiNet_SEBlock_LN(nn.Module):
    def __init__(self, num_defect_tp):
        super(BSL_MultiNet_SEBlock_LN, self).__init__()

        self.pool = nn.AvgPool2d(2, 2)

        self.num_defect_tp = num_defect_tp

        sz = 245760
        sz = 20480 # 256
        #sz = 23040 # 224 
        self.fc1 = nn.Sequential(nn.Dropout(0.7), 
                SEBlock(sz), nn.Linear(sz, 16))

        # if output only 1 node, add one more layer 
        self.fc2 = nn.Linear(16, num_defect_tp)

    def forward(self, m_merge):
        m_merge = self.pool(m_merge)
        x = m_merge.view(m_merge.size(0), -1) #keep batch size
        x = self.fc1(x)
        x = self.fc2(x)
        x = F.sigmoid(x)
        return x
    
class SEBlock(nn.Module):
    def __init__(self, in_ch, r=8):
        super(SEBlock, self).__init__()

        self.linear_1 = nn.Linear(in_ch, in_ch//r)
        self.linear_2 = nn.Linear(in_ch//r, in_ch)

    def forward(self, x):
        input_x = x

        x = F.relu(self.linear_1(x), inplace=True)
        x = self.linear_2(x)
        x = F.sigmoid(x)

        x = input_x * x

        return x

# -------------------------------------- 

from sklearn.model_selection import StratifiedShuffleSplit
from copy import deepcopy

def get_StratifiedShuffleSplit(df, tar_col='target'):
    sss = StratifiedShuffleSplit(n_splits=1, random_state=0)
    for train_index, test_index in sss.split(list(df[tar_col]), list(df[tar_col])):
        train_df = df.iloc[train_index]
        test_df = df.iloc[test_index]
    return train_df, test_df

def balance_hook_df(df, tar_col):
    '''
    Test :
      - (as is) df[tar_col].value_counts()
      - (to be) balance_hook_df(df, tar_col)[tar_col].value_counts()

    Notice this function would be called by agent for dynamically balance
    '''
    res_df = deepcopy(df)
    
    distribution = df[tar_col].value_counts()
    thresh = distribution.max()
    # get times need to be double 
    distribution = (-distribution + thresh)/distribution    

    for i in distribution.index:
        # need to augment-times, if tmp_ind == 1-> need double the dataset
        tmp_ind = distribution.loc[i]
        while tmp_ind > 0:
            res_df = res_df.append(df[df[tar_col]==i], 
                                   ignore_index=True)
            tmp_ind -=1
    res_df.index = range(len(res_df))
    return res_df

# ==================================================
