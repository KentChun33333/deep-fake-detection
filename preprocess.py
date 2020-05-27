
import pandas as pd
import numpy as np
from tqdm import tqdm
import multiprocessing
from functools import partial
import cv2, os, pickle, json, torch 
from mtcnn_net import MTCNN
from video_sampler import FrameSampler
import argparse 
    
 
class FaceSeqExtrator0320():
    def __init__(self, video_path, 
                 mtcnn_model, 
                 save_dir, 
                 margin=20, 
                 frame_sampler_para={'mode': 'one'}, 
                 outimg_size=(240, 240), 
                 mctnn_size = (900, 900),
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
        
        self.mctnn_size = mctnn_size
        self.out_face_size = outimg_size
        self.save_dir = save_dir
        
    def get_images(self, frame_id_list: 'list')-> 'img':
        '''
        Des:
          - giving the frame_id_list for extracting the frames in vedio, then to have mtcnn 
          - to detect the face with its face-lamdmark.
        '''

        save_pth = self.get_output_save_pth()
        if os.path.exists(save_pth):
            return None
        
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
        except Exception as e :
            print(e)
            return None
        
        batch_frame = np.array(batch_frame)
        
        batch_boxes, batch_probs, land_mark = self.mtcnn_model.detect(batch_frame, landmarks=True)

        try:            
            res = {}
            top_k = [ arr.argsort()[-3:][::-1] for arr in batch_probs]
            frames = []      
            for batch_id in range(len(frame_id_list)):
                sg_frame = batch_frame[batch_id]
                face_frames = []
                # how many faces in this frame 

                for face_id in range(len(top_k[batch_id])):
                    face_boxes = batch_boxes[batch_id][face_id]
                    
                    crop_face = self.get_crop_img(sg_frame, face_boxes)
                    face_frames.append(crop_face)
                    if self.show:
                        plt.imshow(crop_face)
                        plt.title(f'frame-id: {batch_id}, face-id:{face_id}, prob:{round(batch_probs[batch_id][face_id], 2)}')
                        plt.show()
                # 
                frames.append(np.array(face_frames))

            pickle.dump(np.array(frames), open(save_pth, 'wb'))
            return None 
        except Exception as e :
            print(e)
            return None


    def get_output_save_pth(self):

        SAVE_dir = self.save_dir
        
        if not os.path.exists(SAVE_dir):
            os.makedirs(SAVE_dir)
        save_pth = os.path.join(SAVE_dir, 
            os.path.basename(self.video_path).split('.')[0]+'.obj')

        return save_pth

    
    def get_crop_img(self, img, bbox):
        y1, x1, y2, x2 = bbox
        x_margin = self.margin
        y_margin = self.margin
        crop_face = img[int(max(x1-x_margin, 0)):int(x2+x_margin), 
                        int(max(y1-y_margin, 0)):int(y2+y_margin), :]
        crop_face = make_square_image(crop_face)
        return cv2.resize(crop_face, self.out_face_size)
            
    def get_samples_by_frame_sampler(self):
        res = {}
        # if res == {} it means there are at least one image cant defect the face 
        temp_try = 0 
        while res=={} and temp_try < 2:
            frame_ids = self.frame_sampler.run()
            print('frame sample ids', frame_ids)

            res = self.get_images(frame_ids)
            temp_try+=1
        if res=={}:
            # print('None')
            return None # pytorch will handle
        return res 

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


def main_func(path, mtcnn_model, save_dir, 
    margin=20, frame_sampler_para={'mode': 'seq', 'seq_size':30, 'gap':3},
    outimg_size=(240, 240), mctnn_size = (900, 900)
    ):
    agent = FaceSeqExtrator0320(path, 
             mtcnn_model=mtcnn_model, 
             save_dir=save_dir, 
             margin=margin, 
             frame_sampler_para= frame_sampler_para, 
             outimg_size=outimg_size, 
             mctnn_size = mctnn_size
             )

    return agent.get_samples_by_frame_sampler()

def make_square_image(img):
    h, w = img.shape[:2]
    size = max(h, w)
    t = 0
    b = size - h
    l = 0
    r = size - w
    return cv2.copyMakeBorder(img, t, b, l, r, cv2.BORDER_CONSTANT, value=0)
    
def get_mtcnn(dir_path, image_size, margin):
    mtcnn = MTCNN(image_size=image_size, margin=margin, pretrained=False, device='cuda:2')
    onet_state_dict = f'{dir_path}/onet.pt'
    pnet_state_dict = f'{dir_path}/pnet.pt'
    rnet_state_dict = f'{dir_path}/rnet.pt'
    mtcnn.onet.load_state_dict(torch.load(onet_state_dict))
    mtcnn.pnet.load_state_dict(torch.load(pnet_state_dict))
    mtcnn.rnet.load_state_dict(torch.load(rnet_state_dict))
    return mtcnn

if __name__ == '__main__':
    
    # 1. get the para
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--main_dir', default='/train_data')
    parser.add_argument('-n', '--mctnn_pretrain_dir', default='/mtcnn')
    parser.add_argument('-o', '--mtcnn_img_size', default=900)
    parser.add_argument('-p', '--mtcnn_margin', default=20)
    parser.add_argument('-q', '--multiprocessing', default=False)
    parser.add_argument('-r', '--save_dir', default='/train_preprocess')
    args = parser.parse_args()

    # 2. get df with path/vid
    df = get_multi_folder_to_df([os.path.join(args.main_dir, i) for i in os.listdir(args.main_dir)])

    # 3. prepare model and process 
    mtcnn = get_mtcnn(args.mctnn_pretrain_dir, args.mtcnn_img_size,args.mtcnn_margin)
    multi_input_list = list(df['video_pth'])
    
    # 4. running 
    res_df = pd.DataFrame()

    if args.multiprocessing:
        multiprocessing.set_start_method('spawn', force=True)    
        with multiprocessing.Pool(processes=2) as pool:
            with tqdm(total=len(multi_input_list)) as pbar:
                for res_temp in tqdm(pool.imap_unordered(partial(
                    main_func, 
                    mtcnn_model = mtcnn, 
                    save_dir=args.save_dir), multi_input_list)): 
                    pbar.update()
                    res_df = res_df.append(res_temp)
     
    else:
        for pth in tqdm(multi_input_list):
            res_temp = main_func(pth, mtcnn, args.save_dir)
            res_df = res_df.append(res_temp)
    # save meta-data
    res_df.to_csv('preprocess_res_df.csv', index=False)