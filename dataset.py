
import os, pickle
from torch.utils.data import DataLoader, Dataset
import numpy as np 
import torch 

# Strategy to gen the batch data under preprocess-pipe 
# the preproess-pipe get the 12 frames in one video with max-to-top3 confident faces in single frame.
# the frame is equal gaped with 3 

       
class Ka2020DeepFackSeq(Dataset):
    def __init__(self, regist_df, phase='train'):
        self.regist_df = regist_df
        self.phase = phase 
        self.fnames = self.regist_df.index.tolist()
            
        # self.augment = Aug_Ka2020DeepFack0204()

    def load_one_face_seq(self, pth):
        obj = pickle.load(open(pth, 'rb'))
        if len(obj.shape)==5:
            # [frame, face, H, W, C]
            # chose best 
            return obj[:, 0, :, :, :]
        else:
            # due to there are inconsistant number of face in seq-frame
            # so we only pick one face in frame 
            obj = np.array([i[0] for i in obj ])
            return obj 

    def __getitem__(self, row_id):
        res = {}
        frames = self.load_one_face_seq(self.regist_df['save_pth'].iloc[row_id])
        # normalization 
        if frames.max() > 1:
            frames = frames/ frames.max()

        res['target'] = self.regist_df.iloc[row_id]['target']
        res['original'] = self.regist_df['original'].iloc[row_id]             
        res['file_name'] = self.regist_df['save_pth'].iloc[row_id]
        
        res['frames'] = frames
    
        return res 
    
    def __len__(self):
        return len(self.fnames)

    def my_collate(self, batch):
        # batch contains a list of tuples of structure (sequence, target)
        res = {}
        # now is numpy
        def pack_face_dim(x, max_dim_face):
            d_frame, d_face, w, h, c = x.shape
            seq_tensor = torch.autograd.Variable(torch.zeros((d_frame, max_dim_face, w, h, c)))
            for frame_id in range(d_frame):
                for face_id in range(d_face):
                    seq_tensor[frame_id, face_id, :, :, :] = torch.Tensor(x[frame_id, face_id, :, :, :])
            return seq_tensor
        
        batch = [i for i in batch if i]


        for col in ['target', 'frames']:
            obj = [item[col] for item in batch]
            #print([i.shape for i in obj])
            res[col] = torch.Tensor(np.stack(obj)) 
                
        for col in ['original', 'file_name']:
            res[col] = [item[col] for item in batch]

                
            # max_dim_face = max([item['frames'].shape[1]  for item in batch])
            # res['frames'] = torch.stack([pack_face_dim(x['frames'], max_dim_face) for x in batch])

        res['batch_idx'] = torch.Tensor([i for i in range(len(res['file_name']))])
        return res