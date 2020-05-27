
from imgaug import augmenters as iaa
from copy import deepcopy

from skimage.io import imshow
import pandas as pd 
import numpy as np 
import cv2, os, PIL, torchvision, pickle, json
from matplotlib import pyplot as plt
# import torch
import torch
from collections import defaultdict
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
#from facenet_pytorch import MTCNN, InceptionResnetV1
#import facenet_pytorch



class Aug2_post():
  def __init__(self):

    self.aug = iaa.Sequential([
        iaa.Sometimes(0.75, iaa.GammaContrast((0.8, 1.25)) ),
        iaa.Fliplr(0.5),
        iaa.Sometimes(0.75, iaa.Affine(rotate=(-15, 15)) ),
    ])

  def __call__(self, img):
    N, W, H, C = img.shape
    dstack_imgs = np.dstack([img[i] for i in range(N)])
    aug_images = self.aug.augment_image(dstack_imgs)
    aug_images = np.array(np.dsplit(aug_images, N))
    if np.max(aug_images) >1:
        aug_images = aug_images/255.
    return aug_images

class Ka2020DeepFackNoise(Dataset):
    def __init__(self, regist_df, frame_sampler_para, outimg_size, phase='train'):
        self.regist_df = regist_df
        # col : video_path and target (0 or 1)
        
        self.mean =   (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)
        self.fnames = self.regist_df.index.tolist()
        # self.ffd_noise_extracotr  = FFDnetSmoother()
        self.phase = phase 
        self.frame_sampler_para = frame_sampler_para
        self.outimg_size = outimg_size
        self.augmentor = Aug2_post()

    def __getitem__(self, row_id):
        video_pth = self.regist_df['video_pth'].iloc[row_id]
        
        extractor = FrameExtrator(video_pth, self.outimg_size, self.frame_sampler_para)   

        res = extractor.get_samples_by_frame_sampler()
        # since frame_extrator retunr NHWC, wherer N is the frames from 1 vid
        res['frames'] = self.augmentor(res['frames'])[0] # here return single img

        try:

            if self.phase =='train':
                res['target'] = self.regist_df.iloc[row_id]['target']
            return res 
        except:
            # print(video_pth) 
            return self.__getitem__(np.random.randint(len(self.fnames)))

    def __len__(self):
        return len(self.fnames)

    def my_collate(self, data):

        res = {}
        # now is numpy

        batch = [i for i in data if i]
        res['target'] = torch.Tensor([item['target'] for item in batch])
        img = np.array([item['frames'] for item in batch])

        # adapt ffdnet 
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

        # img = np.float32(img/255.)
        res['noise_sigma'] = Variable(torch.FloatTensor(
                              [35/255.]*sh_im[0]))
        res['frames'] = torch.Tensor(img)
        # need to process a little bit in pylightening module ... not a good pratice xxx 
        return res 
    
class FrameExtrator():
    def __init__(self, video_path, img_size, frame_sampler_para):
        self.video_path = video_path
        self.video_obj = cv2.VideoCapture(video_path) 
        self.fps = self.video_obj.get(cv2.CAP_PROP_FPS)      # OpenCV2 version 2 used "CV_CAP_PROP_FPS"
        self.frame_count = int(self.video_obj.get(cv2.CAP_PROP_FRAME_COUNT))
        # frame_sampler
        self.frame_sampler = FrameSampler(self.frame_count, self.fps, frame_sampler_para)
        self.frame_sampler_para= frame_sampler_para
        self.img_size = img_size

    def get_images(self, frame_id_list: 'list')-> 'img':
        '''
        Des:
          - giving the frame_id_list for extracting the frames in vedio, then to have mtcnn 
          - to detect the face with its face-lamdmark.
        '''

        # prepare
        batch_frame = []
        res = {}

        try:
            for frame_id in frame_id_list:
                self.video_obj.set(1, frame_id)
                ret, frame = self.video_obj.read()
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_resize = cv2.resize(frame, self.img_size)#.reshape(*self.img_size, 3)
                frame_resize = np.swapaxes(frame_resize,0,2)
                batch_frame.append(frame_resize)
            self.video_obj.release()

            res['frames'] = np.array(batch_frame)
            res['frame_ids'] = frame_id_list
        except Exception as e:
            print(e)

        return res

    def get_samples_by_frame_sampler(self):
        res = {}
        # if res == {} it means there are at least one image cant defect the face 
        temp_try = 0 
        while res=={} and temp_try <5:
            frame_ids = self.frame_sampler.run()
            # print('frame sample ids', frame_ids)
            res = self.get_images(frame_ids)
            # print(res)
            temp_try+=1
        if res=={}:
            return None # pytorch will handle
        return res 


class FrameSampler():
    def __init__(self, max_frame_count, fps, sampler_para:'dict'):
        self.max_frame_count = max_frame_count
        self.fps = max(0, min(int(fps), max_frame_count)-1)

        self.regist_dict = {
            'one'     : self.get_random_one_img, 
            'multi'   : self.get_random_N_img, 
            'one_pair': self.get_one_pair_img, 
            'seq'     : self.get_seq_img,
            'N_pair'  : self.get_N_pair_img, 
        }
        
        self.main_func = self.regist_dict[sampler_para['mode']]
        #del sampler_para['mode']
        self.para_dict = sampler_para


        # while self.fps > (self.max_frame_count-1) and self.max_frame_count>2:
        #     self.fps = max(1, int(self.fps/2))
    
    def get_random_one_img(self, **kwargs):
        return np.random.choice(self.max_frame_count+1, size=1)
    
    def get_random_N_img(self, N, **kwargs):
        return np.random.choice(self.max_frame_count+1, size=N)
    
    def get_one_pair_img(self, **kwargs):

        img_start = np.random.choice(self.max_frame_count-self.fps, size=1)
        return [int(img_start), int(img_start+self.fps)]
    
    def get_seq_img(self, seq_size, **kwargs):
        img_start = np.random.choice(self.max_frame_count-seq_size, size=1)[0]
        return [img_start+i for i in range(seq_size)]
    
    def get_N_pair_img(self,  **kwargs):
        pass  # N pair ~ one Pair 
    def run(self):
        return self.main_func(**self.para_dict)
