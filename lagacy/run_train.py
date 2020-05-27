
# Get the regist df

from model_classify import * 
from model_mtcnn_defect import *
from ka_dataset import *
from agent import *

import json
from sklearn.model_selection import StratifiedShuffleSplit, GroupShuffleSplit
from sklearn.model_selection import GroupShuffleSplit
from copy import deepcopy

def get_StratifiedShuffleSplit(df, tar_col='target'):
    sss = StratifiedShuffleSplit(n_splits=1, random_state=0)
    for train_index, test_index in sss.split(list(df[tar_col]), list(df[tar_col])):
        train_df = df.iloc[train_index]
        test_df = df.iloc[test_index]
    return train_df, test_df

def get_GroupShuffleSplit(df, col='chunk'):
    gss = GroupShuffleSplit(n_splits=1, train_size=.8, random_state=42)
    for train_index, test_index in gss.split(list(df[col]), list(df[col]), list(df[col])):
        train_df = df.iloc[train_index]
        test_df = df.iloc[test_index]
    return train_df, test_df

def balance_target_df(df, col):
    res= []
    min_value = df[col].value_counts().min()
    for v in list(set(df[col])):
        print(v)
        temp_df = df[df[col]==v]
        res.append(temp_df.sample(n=min_value))
    return pd.concat(res, ignore_index=True)

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
    df['chunk'] = int(vid_folder.split('_')[-1])

    # this is faster loader defined by fast_defection
    SAVE_dir = '/test/deepfack_detection_challenge_sample/save_face/multi/'
    df['save_10_pth'] = df['video_pth'].apply(lambda x: os.path.join(SAVE_dir, 
            os.path.basename(x).split('.')[0]+'.obj'))    

    df['save_10_pth_true'] = df['save_10_pth'].apply(lambda x: os.path.exists(x) ) 
    df = df[df['save_10_pth_true']==True]
    return df 

def get_multi_folder_to_df(vid_folder_list):
    res = []
    for vid_folder in vid_folder_list:
        res.append(get_single_folder_df(vid_folder))
    return pd.concat(res, ignore_index=True)


if __name__ =='__main__':
    para_flag = Namespace()
    para_flag.batch_size = 4
    para_flag.mctnn_size = (900, 900) # important
    para_flag.outimg_size = (224, 224)
    para_flag.learning_rate= 1e-4
    para_flag.frame_sampler_para = {'mode' : 'one_pair' }    

    mtcnn = MTCNN(image_size=900, margin=0, pretrained=False, device='cuda:2')
    onet_state_dict = '/test/torch_pretrain/onet.pt'
    pnet_state_dict = '/test/torch_pretrain/pnet.pt'
    rnet_state_dict = '/test/torch_pretrain/rnet.pt'
    mtcnn.onet.load_state_dict(torch.load(onet_state_dict))
    mtcnn.pnet.load_state_dict(torch.load(pnet_state_dict))
    mtcnn.rnet.load_state_dict(torch.load(rnet_state_dict))

    # model = PairClasssifer_DeepFaceSys(para_flag)   
    model = PairClasssifer_DeepFaceMultiSys(para_flag) 

    main_dir= '/test/deepfack_detection_challenge_sample/train_00/'
    df = get_multi_folder_to_df([os.path.join(main_dir, i) for i in os.listdir(main_dir)])    

    df['target'] = df['target'].apply(lambda x: 1 if x=='FAKE' else 0)    

    df = balance_target_df(df, 'target')
    
    train_df = df[df['chunk']<=40]
    valid_df = df[df['chunk']>40]
    train_df, test_df = get_GroupShuffleSplit(train_df)    

    print(len(train_df), len(valid_df), len(test_df))
        
    #train_df = balance_hook_df(train_df, 'target')
    valid_df.index = range(len(valid_df))
    test_df.index = range(len(test_df))    
    
    print(train_df.target.value_counts())
    model.build_dataset(train_df, valid_df, test_df, mtcnn)    
    

    trainer = Trainer(gpus=[1], accumulate_grad_batches=2, gradient_clip=2, 
        max_nb_epochs=12)    
    trainer.fit(model) 


# for pytorch -lightning 
# torch.save(
# torch.load('/kaggle/input/model-pt2/ep5.pt',map_location='cpu')['state_dict'], 
# 'ep5_small.pt')

class OutputModelWrap(nn.Module):
    def __init__(self):
        super(OutputModelWrap, self).__init__()
        self.model = PairImgClasssifer_DeepFace()
        
    def forward(self, x, y):
        return self.model(x,y)


# ----------
# avoid chagne the image asperatio 
def isotropically_resize_image(img, size, resample=cv2.INTER_AREA):
    h, w = img.shape[:2]
    if w > h:
        h = h * size // w
        w = size
    else:
        w = w * size // h
        h = size

    resized = cv2.resize(img, (w, h), interpolation=resample)
    return resized


def make_square_image(img):
    h, w = img.shape[:2]
    size = max(h, w)
    t = 0
    b = size - h
    l = 0
    r = size - w
    return cv2.copyMakeBorder(img, t, b, l, r, cv2.BORDER_CONSTANT, value=0)