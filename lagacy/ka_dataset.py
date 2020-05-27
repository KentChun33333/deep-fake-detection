
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

from torch.utils.data import DataLoader, Dataset
#from facenet_pytorch import MTCNN, InceptionResnetV1
#import facenet_pytorch
import kornia


        
class Ka2020DeepFackSeq(Dataset):
    def __init__(self, regist_df, phase='train'):
        self.regist_df = regist_df
        self.phase = phase 
        
        if self.phase=='train':
            # the key is to align Real and Fake by Original and 
            # make sure in every batch, same original Real and Fake is pumping 
            # align the original would benifit to reduce the noice-information and make the neural net converge fast
            # 
            self.true_df = regist_df[regist_df.target==0]
            self.true_df = self.true_df.sort_values('original')
            self.true_df.index = range(len(self.true_df))
        
            self.fake_df = regist_df[regist_df.target!=0]
            self.fake_df = self.fake_df.sort_values('original')
            self.fake_df.index = range(len(self.fake_df))

            assert list(self.fake_df.original) == list(self.true_df.original)
            assert len( self.fake_df) == len( self.true_df)

        # check the dast-defect_preprocess for 
            self.fnames = self.true_df.index.tolist()
        else:
            self.fnames = self.regist_df.index.tolist()
            
        self.augment = Aug_Ka2020DeepFack0204()
        
    def __getitem__(self, row_id):
        res = {}
        if self.phase=='train':
            frames_true = pickle.load(open(self.true_df['save_10_pth'].iloc[row_id], 'rb'))
            frame_fake = pickle.load(open(self.fake_df['save_10_pth'].iloc[row_id], 'rb'))
            frames_true = np.array([i[:1, :, :, :] for i in frames_true])
            frame_fake = np.array([i[:1, :, :, :] for i in frame_fake])

            # why frame true and face would have different crop?
            # since ... shit ....the data preprocess need to be different 
            # the face recognition should be base on only true vid and 
            # if origion is the same then we use the correponding bbox 
            #frame_id, t_face_id, H, W, C = frames_true.shape
            #frame_id, f_face_id, H, W, C = frame_fake.shape
            #t_face_id = min(t_face_id, f_face_id)
            
            if np.max(frames_true) > 1.1: frames_true = frames_true/255.
            if np.max(frame_fake) > 1.1: frame_fake = frame_fake/255.

            res['target'] = [1, 0]
            res['frames'] = [self.augment(frames_true), self.augment(frame_fake)]
            res['original'] = [ self.true_df['original'].iloc[row_id], 
                              self.fake_df['original'].iloc[row_id] ]
            
            res['file_name'] = [self.true_df['save_10_pth'].iloc[row_id], 
                                self.fake_df['save_10_pth'].iloc[row_id] ]
            
            temp = frames_true[0][0]

        else:
            frames =  pickle.load(open(self.regist_df['save_10_pth'].iloc[row_id], 'rb'))
            # frames = [frame_1, frame_2 ... etc ], in frame1 it has dimension of [face, H, W, C]
            min_face = min([i.shape[0] for i in frames])
            frames = np.array([i[:min_face, :, :, :] for i in frames[8:10]])
#             print(f'frames shape in dataset {frames.shape}')
            if frames.max() > 1:
                frames = frames/ frames.max()

            res['target'] = self.regist_df.iloc[row_id]['target']
            
            res['original'] = self.regist_df['original'].iloc[row_id]             
            res['file_name'] = self.regist_df['save_10_pth'].iloc[row_id]
            
            res['frames'] = frames
            temp = res['frames'][0][0]
        
        # print(temp.shape)
#         if np.any(np.isnan(temp)): 
#             print('nan')
#             return None

        return res 
    
    def check_nan(self, x):
        if np.any(np.isnan(x)): 
            return None

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
        if self.phase=='train':
            for col in ['target']:
                res[col] = torch.Tensor(
                    [item[col][0] for item in batch] + [item[col][1] for item in batch])
            
            for col in ['original', 'file_name']:
                res[col] = [item[col][0] for item in batch] + [item[col][1] for item in batch]
                
            max_dim_face = max(
                max([item['frames'][0].shape[1]  for item in batch]), 
                max([item['frames'][1].shape[1]  for item in batch]))
            
            res['frames'] = torch.stack([pack_face_dim(x['frames'][0], max_dim_face) for x in batch]+
                                        [pack_face_dim(x['frames'][1], max_dim_face) for x in batch])
            

        else:
            for col in ['target']:
                res[col] = torch.Tensor([item[col] for item in batch])
                
            for col in ['original', 'file_name']:
                res[col] = [item[col] for item in batch]

                
            max_dim_face = max([item['frames'].shape[1]  for item in batch])
            res['frames'] = torch.stack([pack_face_dim(x['frames'], max_dim_face) for x in batch])
        # item here [frame, faces]
        # print(batch[0]['frames'].shape)
        res['batch_idx'] = torch.Tensor([i for i in range(len(res['file_name']))])
        return res

    
class Ka2020DeeFackFrameSeq(Ka2020DeepFackSeq):
    def __getitem__(self, row_id):
        res = {}
        if self.phase=='train':
            frames_true = pickle.load(open(self.true_df['save_10_pth'].iloc[row_id], 'rb'))
            frame_fake = pickle.load(open(self.fake_df['save_10_pth'].iloc[row_id], 'rb'))

            #if np.max(frames_true) > 1.1: frames_true = frames_true/255.
            #if np.max(frame_fake) > 1.1: frame_fake = frame_fake/255.

            res['target'] = [1, 0]
            res['frames'] = [self.augment(frames_true[7:10, :, :, : ]), self.augment(frame_fake[7:10, :, :, : ]) ]
            res['original'] = [ self.true_df['original'].iloc[row_id], 
                              self.fake_df['original'].iloc[row_id] ]
            
            res['file_name'] = [self.true_df['save_10_pth'].iloc[row_id], 
                                self.fake_df['save_10_pth'].iloc[row_id] ]
            

        else:
            frames =  pickle.load(open(self.regist_df['save_10_pth'].iloc[row_id], 'rb'))
            # frames = [frame_1, frame_2 ... etc ], in frame1 it has dimension of [face, H, W, C]

            res['target'] = self.regist_df.iloc[row_id]['target']
            
            res['original'] = self.regist_df['original'].iloc[row_id]             
            res['file_name'] = self.regist_df['save_10_pth'].iloc[row_id]
            
            res['frames'] = self.augment(frames[8:10, :, :, : ])
#             print('res frame max : ', res['frames'].max())
        
        return res 
    
    def my_collate(self, batch):
        # batch contains a list of tuples of structure (sequence, target)
        res = {}

        batch = [i for i in batch if i]
        if self.phase=='train':
            for col in ['target', 'frames']:
                res[col] = torch.Tensor(
                    [item[col][0] for item in batch] + [item[col][1] for item in batch])
            
            for col in ['original', 'file_name']:
                res[col] = [item[col][0] for item in batch] + [item[col][1] for item in batch]

        else:
            for col in ['target', 'frames']:
                res[col] = torch.Tensor([item[col] for item in batch])
                
            for col in ['original', 'file_name']:
                res[col] = [item[col] for item in batch]
        res['batch_idx'] = torch.Tensor([i for i in range(len(res['file_name']))])
        return res
    

class Ka2020DeepFack(Dataset):
    def __init__(self, regist_df, frame_sampler_para, face_defector_mctnn, 
        mctnn_size, outimg_size, phase='train'):
        self.regist_df = regist_df
        # col : video_path and target (0 or 1)

        # self.transforms = get_transforms(phase, self.mean, self.std) 
        # get_transforms(phase, mean, std)
        
        self.fnames = self.regist_df.index.tolist()
        self.phase = phase 
        
        self.face_defector_mctnn = face_defector_mctnn
        self.frame_sampler_para = frame_sampler_para
        self.mctnn_size = mctnn_size
        self.outimg_size = outimg_size
        self.augment_agent = Aug_Ka2020DeepFack0204()
        

    def __getitem__(self, row_id):
        video_pth = self.regist_df['video_pth'].iloc[row_id]
        
        face_agent = FrameExtrator(video_path=video_pth, 
                                   mtcnn_model=self.face_defector_mctnn, 
                                   frame_sampler_para=self.frame_sampler_para, 
                                   mctnn_size = self.mctnn_size, 
                                   outimg_size = self.outimg_size,
                                   show=False)   

        res = face_agent.get_samples_by_frame_sampler()
        try:
            if np.any(np.isnan(res['frames'])):
                # print('nan')
                return self.__getitem__(np.random.randint(len(self.fnames)))

            if np.any(res['frames']<0) or np.any(res['frames']>1):
                # print('out of boundary')
                return self.__getitem__(np.random.randint(len(self.fnames)))

            if self.phase =='train':
                res['target'] = self.regist_df.iloc[row_id]['target']
            return res 
        except:
            # print(video_pth) 
            return self.__getitem__(np.random.randint(len(self.fnames)))

    def __len__(self):
        return len(self.fnames)

class Ka2020DeepFackMulti(Dataset):
    def __init__(self, regist_df, frame_sampler_para, face_defector_mctnn, 
        mctnn_size = (500, 500), outimg_size=(256, 256) , phase='train'):
        self.regist_df = regist_df
        # col : video_path and target (0 or 1)
        
        self.mean =   (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)
        # self.transforms = get_transforms(phase, self.mean, self.std) 
        # get_transforms(phase, mean, std)
        
        self.fnames = self.regist_df.index.tolist()
        self.phase = phase 
        
        self.face_defector_mctnn = face_defector_mctnn
        self.frame_sampler_para = frame_sampler_para
        self.mctnn_size = mctnn_size
        self.outimg_size = outimg_size
        

    def __getitem__(self, row_id):
        video_pth = self.regist_df['video_pth'].iloc[row_id]
        save_pth = self.regist_df['save_10_pth'].iloc[row_id]
        res = {}

        frames = pickle.load(open(save_pth, 'rb'))
        if np.max(frames) > 1.1:
            frames /= 255.
        res['frames'] = frames
        
        temp = res['frames'][0][0]

        # print(temp.shape)
        if np.any(np.isnan(temp)): return None
        if np.any(temp<0) or np.any(temp>1): return None

        if self.phase =='train':
            res['target'] = self.regist_df.iloc[row_id]['target']
        return res 
        # face_agent = FaceExtractorMulti(video_path=video_pth, 
        #                            mtcnn_model=self.face_defector_mctnn, 
        #                            frame_sampler_para=self.frame_sampler_para, 
        #                            mctnn_size = self.mctnn_size, 
        #                            outimg_size = self.outimg_size,
        #                            show=False)   
        # res = face_agent.get_samples_by_frame_sampler()
        

    def __len__(self):
        return len(self.fnames)
    ## a simple custom collate function, just to show the idea
    ## `batch` is a list of tuple where first element is image tensor and
    ## second element is corresponding label

    #def my_collate(batch):
    #    data = [item[0] for item in batch]  # just form a list of tensor
    #
    #    target = [item[1] for item in batch]
    #    target = torch.LongTensor(target)
    #    return [data, target]
    def my_collate(self, batch):
        # batch contains a list of tuples of structure (sequence, target)
        res = {}
        # now is numpy

        batch = [i for i in batch if i]
        res['target'] = torch.Tensor([item['target'] for item in batch])
        # item here [frame, faces]
        # print(batch[0]['frames'].shape)
        max_dim_face = max([item['frames'].shape[1]  for item in batch])

        def pack_face_dim(x):
            d_frame, d_face, w, h, c = x.shape
            seq_tensor = torch.autograd.Variable(torch.zeros((d_frame, max_dim_face, w, h, c)))
            for frame_id in range(d_frame):
                for face_id in range(d_face):
                    seq_tensor[frame_id, face_id, :, :, :] = torch.Tensor(x[frame_id, face_id, :, :, :])
            return seq_tensor

        res['frames'] = torch.stack([pack_face_dim(x['frames']) for x in batch])
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
        return [int(0), int(img_start+self.fps)]
    
    def get_seq_img(self, seq_size, gap, **kwargs):
        return [0+i*gap for i in range(seq_size)]
    
    def get_N_pair_img(self,  **kwargs):
        pass  # N pair ~ one Pair 
    def run(self):
        return self.main_func(**self.para_dict)
             
class FaceExtractor():
    def __init__(self, video_path, 
                 mtcnn_model, 
                 margin=0.3, 
                 frame_sampler_para={'mode': 'one'}, 
                 outimg_size=(256, 256), 
                 mctnn_size = (500, 500),
                 show=False):
        '''
        There are one 4-templat of frame_sampler_para 
          - {'mode' : 'seq' , 'seq_size':10}
          - {'mode' : 'multi' , 'N':17 }
          - {'mode' : 'one' }
          - {'mode' : 'one_pair' }        
        '''

        self.video_path = video_path

        self.video_obj = cv2.VideoCapture(video_path) 
        
        self.fps = self.video_obj.get(cv2.CAP_PROP_FPS)      # OpenCV2 version 2 used "CV_CAP_PROP_FPS"
        self.frame_count = int(self.video_obj.get(cv2.CAP_PROP_FRAME_COUNT))
        # frame_sampler
        self.frame_sampler = FrameSampler(self.frame_count, self.fps, frame_sampler_para)
        self.frame_sampler_para= frame_sampler_para

        self.mtcnn_model = mtcnn_model
        self.show = show
        self.margin = margin
        # 
        self.mctnn_size = mctnn_size
        self.out_face_size = outimg_size

    def get_images(self, frame_id_list: 'list')-> 'img':
        '''
        Des:
          - giving the frame_id_list for extracting the frames in vedio, then to have mtcnn 
          - to detect the face with its face-lamdmark.
        '''

        # prepare
        batch_frame = []
        try:
            for frame_id in frame_id_list:
                self.video_obj.set(1, frame_id)
                ret, frame = self.video_obj.read()
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_resize = cv2.resize(frame, self.mctnn_size).reshape(*self.mctnn_size, 3)
                batch_frame.append(frame_resize)
            self.video_obj.release()
        except:
            return {}
        
        batch_frame = np.array(batch_frame)
        if self.mtcnn_model is None:
            res = {}
            res['frames'] = batch_frame
            res['frame_ids'] = frame_id_list
            return res 

        batch_boxes, batch_probs, land_mark = self.mtcnn_model.detect(batch_frame, landmarks=True)

        # from blazeface import BlazeFace
        # net = BlazeFace()
        # net.load_weights("blazeface.pth")
        # net.load_anchors("anchors.npy")        

        # # Optionally change the thresholds:
        # net.min_score_thresh = 0.75
        # net.min_suppression_threshold = 0.3
        # res = net(batch_frame)
        # print('face_extractor')
        # print(res)
        # GET MAX PROB FACE
        try:
            max_idx = np.argmax(batch_probs, axis=1)
            max_batch_probs = set(np.max(batch_probs, axis=1))
        except:
            return {}
                # There will be some image cant extract face 
        if None in max_batch_probs:
            # print('None in batch')
            return {}
        
        res = {'frames': [], 'frame_ids': [], 'land_marks':[] }
        
        for batch_id in range(len(max_idx)):
            sg_landmark = land_mark[batch_id, max_idx[batch_id] , :, :]
            face_boxes     = batch_boxes[batch_id, max_idx[batch_id], :]

            sg_frame = batch_frame[batch_id]

            y1, x1, y2, x2 = face_boxes
            
            x_margin = int((x2-x1)*self.margin)
            y_margin = int((y2-y1)*self.margin)
            sg_frame = sg_frame[int(max(x1-x_margin, 0)):int(x2+x_margin), int(max(y1-y_margin, 0)):int(y2+y_margin), :]
            
            sg_frame = cv2.resize(sg_frame, self.out_face_size)#.reshape(*outimg_size, 3)
            if self.show:
                pass 
                # imshow(sg_frame)
                #for i in range(len(sg_landmark)):
                #    sg_landmark[i] -= np.array([y1-y_margin, x1-x_margin])
                #plt.plot(sg_landmark[:, 0], sg_landmark[:, 1], '.')
                #plt.show()
            res['frames'].append(sg_frame)
            res['frame_ids'].append(frame_id_list[batch_id])
            res['land_marks'].append(sg_landmark)
    
        return  res
    
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

class FaceExtractorMulti():
    def __init__(self, video_path, 
                 mtcnn_model, 
                 margin=0.2, 
                 frame_sampler_para={'mode': 'one'}, 
                 outimg_size=(256, 256), 
                 mctnn_size = (500, 500),
                 pre_detect_augmentor = None,
                 post_detect_augmentor = None,
                 show=False):
        '''
        There are one 4-templat of frame_sampler_para 
          - {'mode' : 'seq' , 'seq_size':10}
          - {'mode' : 'multi' , 'N':17 }
          - {'mode' : 'one' }
          - {'mode' : 'one_pair' }        
        '''

        self.video_path = video_path

        self.video_obj = cv2.VideoCapture(video_path) 
        
        self.fps = self.video_obj.get(cv2.CAP_PROP_FPS)      # OpenCV2 version 2 used "CV_CAP_PROP_FPS"
        self.frame_count = int(self.video_obj.get(cv2.CAP_PROP_FRAME_COUNT))
        # frame_sampler
        self.frame_sampler = FrameSampler(self.frame_count, self.fps, frame_sampler_para)
        self.frame_sampler_para= frame_sampler_para

        self.mtcnn_model = mtcnn_model
        self.show = show
        self.margin = margin
        self.post_detect_augmentor = Aug2_post()
        self.pre_detect_augmentor = Aug2_pre()
        # 
        self.mctnn_size = mctnn_size
        self.out_face_size = outimg_size
        
        # print(frame_sampler_para)
        
        
    def get_images(self, frame_id_list: 'list')-> 'img':
        '''
        Des:
          - giving the frame_id_list for extracting the frames in vedio, then to have mtcnn 
          - to detect the face with its face-lamdmark.
        '''

        # prepare
        batch_frame = []
        try:
            for frame_id in frame_id_list:
                self.video_obj.set(1, frame_id)
                ret, frame = self.video_obj.read()
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = make_square_image(frame)
                frame_resize = cv2.resize(frame, self.mctnn_size).reshape(*self.mctnn_size, 3)
                batch_frame.append(frame_resize)
            self.video_obj.release()
        except: return {}
        
        batch_frame = np.array(batch_frame)
        
        if self.pre_detect_augmentor:
            batch_frame = self.pre_detect_augmentor(batch_frame)
        # only use first frame-id to identify the face
        batch_boxes, batch_probs, land_mark = self.mtcnn_model.detect([batch_frame[0]], landmarks=True)
        # if multi-frame, only use first bbox
        # batch_probs Dim is [frame-id, how many face ]
        try:
            batch_boxes = batch_boxes[0] 
            
            res = {}
        
            if self.post_detect_augmentor:
                batch_frame = self.post_detect_augmentor(batch_frame)
            
            # batch_id represents multi-frames_id here   
            frames = []      
            for batch_id in range(len(frame_id_list)):
                sg_frame = batch_frame[batch_id]
            
                face_frames = []
                for face_id in range(len(batch_probs[0])):
                    face_boxes     = batch_boxes[face_id]
                    crop_face = self.get_crop_img(sg_frame, face_boxes)
            
                    if self.show:
                        # imshow(crop_face) ; plt.show()
                        pass
                    face_frames.append(crop_face)
                frames.append(np.array(face_frames))

            res['frames'] = np.array(frames) # frames, how-many-face, HWC
            res['frame_ids'] = frame_id_list
    
            return  res
        except:
            return {}
    
    def get_crop_img(self, img, bbox):
        y1, x1, y2, x2 = bbox
        x_margin = int((x2-x1)*self.margin)
        y_margin = int((y2-y1)*self.margin)
        crop_face = img[int(max(x1-x_margin, 0)):int(x2+x_margin), 
                        int(max(y1-y_margin, 0)):int(y2+y_margin), :]
        crop_face = make_square_image(crop_face)
        return cv2.resize(crop_face, self.out_face_size)
            
    def get_samples_by_frame_sampler(self):
        res = {}
        # if res == {} it means there are at least one image cant defect the face 
        temp_try = 0 
        while res=={} and temp_try < 5:
            frame_ids = self.frame_sampler.run()
            # print('frame sample ids', frame_ids)
            res = self.get_images(frame_ids)
            temp_try+=1
        if res=={}:
            # print('None')
            return None # pytorch will handle
        return res 

class Aug_Ka2020DeepFack0204():
  def __init__(self):

    self.aug = iaa.Sequential([
        #iaa.Sometimes(0.4, iaa.Crop(px=(0, 5))),
        iaa.Resize({"height": 240, "width": 240}), #too small 
        iaa.Sometimes(0.8, iaa.Crop(px=(0, 20))),
        iaa.Fliplr(0.3),
        #iaa.Flipud(0.3),
        #iaa.Affine(rotate=(-5, 5)),
        iaa.GammaContrast((0.8, 1.2)),
        
        #iaa.Sometimes(0.2, iaa.ContrastNormalization((0.75, 1.5))),
    ])

  def __call__(self, img):
#     N, W, H, C = img.shape
#     dstack_imgs = np.dstack([img[i] for i in range(N)])
#     aug_images = self.aug.augment_image(dstack_imgs)
#     aug_images = np.array(np.dsplit(aug_images, N))
    aug_images = img 
    if np.max(aug_images) >1:
        # aug_images = aug_images/255.
        aug_images = aug_images/np.max(aug_images) # this is reported by ka-community that normalize might hinder the result 
    return aug_images



        
class Aug_Ka2020DeepFack0204_test():
  def __init__(self):
      pass 

  def __call__(self, img):
      if np.max(img) >1:
          img = img/255.
      return img

class Aug2_pre():
  def __init__(self):

    self.aug = iaa.Sequential([
        iaa.Fliplr(0.3),
        iaa.Affine(rotate=(-5, 5)),
    ])
    
  def __call__(self, img):
    N, W, H, C = img.shape
    dstack_imgs = np.dstack([img[i] for i in range(N)])
    aug_images = self.aug.augment_image(dstack_imgs)
    aug_images = np.array(np.dsplit(aug_images, N))
    return aug_images

class Aug2_post(Aug_Ka2020DeepFack0204):
  def __init__(self):

    self.aug = iaa.Sequential([
        iaa.GammaContrast((0.7, 1.3)),
        iaa.Sometimes(0.4, iaa.ContrastNormalization((0.75, 1.5))),
    ])

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

def read_meta_data(pth):
    with open(pth) as json_file:
        data = json.load(json_file)
    return data

def get_single_folder_df(vid_folder):
    meta_data = read_meta_data([os.path.join(vid_folder, i) for i in os.listdir(vid_folder) if 'metadata' in i][0])
    df = pd.DataFrame()    
    df['target']  = [meta_data[i]['label'] for i in meta_data.keys()]
    df['original'] =  [meta_data[i]['original'] if 'original' in meta_data[i].keys() else i for i in meta_data.keys() ]
    df['video_pth']  = [os.path.join(vid_folder, i) for i in meta_data.keys()]
    return df 

def get_multi_folder_to_df(vid_folder_list):
    res = []
    for vid_folder in vid_folder_list:
        res.append(get_single_folder_df(vid_folder))
    return pd.concat(res, ignore_index=True)

    
def make_square_image(img):
    h, w = img.shape[:2]
    size = max(h, w)
    t = 0
    b = size - h
    l = 0
    r = size - w
    return cv2.copyMakeBorder(img, t, b, l, r, cv2.BORDER_CONSTANT, value=0)
        