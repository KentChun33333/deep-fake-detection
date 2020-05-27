from model_mtcnn_defect import *

import os, pickle, json
import pandas as pd
import numpy as np
from ka_dataset import *
from tqdm import tqdm
import multiprocessing
from functools import partial

class FaceExtractorMulti_2(FaceExtractorMulti):
    def __init__(self, video_path, 
                 mtcnn_model, 
                 margin=0.25, 
                 frame_sampler_para={'mode': 'seq', 'seq_size':10, 'gap':5}, 
                 outimg_size=(256, 256), 
                 mctnn_size = (480, 480),
                 pre_detect_augmentor = None,
                 post_detect_augmentor = None,
                 show=False):
        super().__init__(video_path, 
                 mtcnn_model, 
                 margin=margin, 
                 frame_sampler_para=frame_sampler_para, 
                 outimg_size=outimg_size, 
                 mctnn_size = mctnn_size,
                 show=False)     
        if not mtcnn_model:
            self.no_defection = True
        else:
            self.no_defection = False
            print('use mtcnn to extract face ')
        
    def get_images(self, frame_id_list: 'list')-> 'img':
        
        SAVE_dir = '/test/deepfack_detection_challenge_sample/save_face/{}/'.format(
            self.frame_sampler_para['mode']+'_seq_12_gap_5_align_face')
        
        if self.no_defection:
            SAVE_dir = '/test/deepfack_detection_challenge_sample/save_face/{}/'.format(
                self.frame_sampler_para['mode']+'_seq_12_gap_5_full_480')
        
        if not os.path.exists(SAVE_dir):
            os.makedirs(SAVE_dir)
        save_name = os.path.join(SAVE_dir, 
            os.path.basename(self.video_path).split('.')[0]+'.obj')

        if os.path.exists(save_name):
            return None

        try:
            batch_frame = []
            for frame_id in frame_id_list:
                self.video_obj.set(1, frame_id)
                ret, frame = self.video_obj.read()
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # keep original HW ratio
                frame = make_square_image(frame)
                frame_resize = cv2.resize(frame, self.mctnn_size).reshape(*self.mctnn_size, 3)
                batch_frame.append(frame_resize)
            self.video_obj.release()
            print(frame_id_list)
        except Exception as e :
            print(e)
            return {}
        
        batch_frame = np.array(batch_frame)
        
        if self.no_defection:
            pickle.dump(np.array(batch_frame), open(save_name, 'wb'))
            return 
            

        # only use first frame-id to identify the face will fail 
        # need 
        batch_boxes, batch_probs, land_mark = self.mtcnn_model.detect(batch_frame, landmarks=True)
        # if multi-frame, only use first bbox
        # batch_probs Dim is [frame-id, how many face ]
        try:
            frames = []      
            for batch_id in range(len(frame_id_list)):
                sg_frame = batch_frame[batch_id]
                face_frames = []
                # how many faces in this frame 
                for face_id in range(len(batch_probs[batch_id])):
                    face_boxes = batch_boxes[batch_id][face_id]
                    crop_face = self.get_crop_img(sg_frame, face_boxes)
                    face_frames.append(crop_face)
                frames.append(np.array(face_frames))

            pickle.dump(frames, open(save_name, 'wb'))
        except Exception as e :
            print(e)
        return {}
    

def make_square_image(img):
    h, w = img.shape[:2]
    size = max(h, w)
    t = 0
    b = size - h
    l = 0
    r = size - w
    return cv2.copyMakeBorder(img, t, b, l, r, cv2.BORDER_CONSTANT, value=0)
    
def get_single_folder_df(vid_folder):

    def read_meta_data(pth):
        with open(pth) as json_file:
            data = json.load(json_file)
        return data

    meta_data = read_meta_data([os.path.join(vid_folder, i) 
                                for i in os.listdir(vid_folder) 
                                if 'metadata' in i][0])
    df = pd.DataFrame()
    df['target']  = [meta_data[i]['label'] for i in meta_data.keys()]
    df['original'] =  [meta_data[i]['original'] if 'original' in meta_data[i].keys() else i for i in meta_data.keys() ]
    df['video_pth']  = [os.path.join(vid_folder, i) for i in meta_data.keys()]
    df['chunk'] = int(vid_folder.split('_')[-1])
    return df 

def get_multi_folder_to_df(vid_folder_list):
    res = []
    for vid_folder in vid_folder_list:
        res.append(get_single_folder_df(vid_folder))
    return pd.concat(res, ignore_index=True)

def multi_proces_func(path, mtcnn_model):
    agent = FaceExtractorMulti_2(path, 
             mtcnn_model=None,                  
             margin=0.26, 
             frame_sampler_para={'mode': 'seq', 'seq_size':10, 'gap':5}, 
             outimg_size=(224, 224), 
             mctnn_size = (480, 480), # output size or mtcnn input size 
             pre_detect_augmentor = None,
             post_detect_augmentor = None,)
    agent.get_samples_by_frame_sampler()

def main():
    mtcnn = MTCNN(image_size=900, margin=0, pretrained=False, device='cuda:2')
    onet_state_dict = '/test/torch_pretrain/onet.pt'
    pnet_state_dict = '/test/torch_pretrain/pnet.pt'
    rnet_state_dict = '/test/torch_pretrain/rnet.pt'
    mtcnn.onet.load_state_dict(torch.load(onet_state_dict))
    mtcnn.pnet.load_state_dict(torch.load(pnet_state_dict))
    mtcnn.rnet.load_state_dict(torch.load(rnet_state_dict))

    # model = PairClasssifer_DeepFaceSys(para_flag)   

    main_dir= '/test/deepfack_detection_challenge_sample/train_00/'
    save_dir = '/test/deepfack_detection_challenge_sample/save_face/'
    
    df = pd.read_csv('original_below_55.csv') # all 

    input_list = list(df['video_pth'])
    # for path in tqdm(list(df['video_pth'])):
    
    multiprocessing.set_start_method('spawn', force=True)    

    res_flag_list = []
    print('Start Reverse Mapping ')
    with multiprocessing.Pool(processes=10) as pool:
        with tqdm(total=len(input_list)) as pbar:
            for temp in tqdm(pool.imap_unordered(partial(
                multi_proces_func, 
                mtcnn_model = None) , input_list)) : 
                pbar.update()
                res_flag_list.append(temp)        

def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    # return the intersection over union value
    return iou
                
                
if __name__ == '__main__':
    main()