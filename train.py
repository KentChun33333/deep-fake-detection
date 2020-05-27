import os, json
import pandas as pd 
import numpy as np 
from sklearn.model_selection import StratifiedShuffleSplit, GroupShuffleSplit
from copy import deepcopy
from dataset import Ka2020DeepFackSeq 
from agent import DeepFace_Seq_Sys
# build config 
import config 
import trainer 
from registor import Registed_Trainer, Registed_Config


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

def select_with_file(df, col):
    ind = df[col].apply(lambda x: os.path.exists(x) ) 
    df = df[ind]
    df.index = range(len(df))
    return df 

  
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
        
def prepare_train_valid(df):


    df = select_with_file(df, 'save_pth')
    print(len(df))
    df['target'] = df['target'].apply(lambda x: 1 if x=='FAKE' else 0)    
    
    train_df = df[df['chunk']<40]
    valid_df = df[df['chunk']>=40]
    
    for i in [train_df, valid_df]:
        i.index = range(len(i))
    
    print('--')
    print('train length: ' , len(train_df))
    print('valid length: ' , len(valid_df))        
    return train_df, valid_df
    
def cli_notice():
    '''just a command line interface'''
    print('-----------------------------------------')
    print('Available configs as follow : ')
    for ind, i in enumerate(Registed_Config.module_dict.keys()):
        print(ind, ' : ' ,i)
    print('-----------------------------------------')
    print('input [config-name-above] + enter to run experiment,')
    config_name = input()
    return config_name

if __name__ == '__main__':

    # get original df 
    main_dir= '/test/deepfack_detection_challenge_sample/train_00/'
    df = get_multi_folder_to_df([os.path.join(main_dir, i) for i in os.listdir(main_dir)]) 
    df['base_name'] = df.video_pth.apply(lambda x : os.path.basename(x))
    # get datafiles 
    dirpth = '/train_preprocess'
    files = [os.path.join(dirpth, i) for i in os.listdir(dirpth)]

    ok_vid = [i.split('/')[-1].replace('.obj', '.mp4') for i in files ]
    ok_df = df[df['base_name'].isin(ok_vid)]
    ok_df['save_pth'] = ok_df['base_name'].apply(
        lambda x: os.path.join(dirpth, x.replace('.mp4', '.obj')))

    print('-- check raw-df ')
    print(ok_df.target.value_counts())
    
    config_name = cli_notice()
    para_flag = Registed_Config(config_name).get_config()

    train_df, valid_df = prepare_train_valid(ok_df)
    print(f'====== Start Training {para_flag.model_name} with Sample {len(train_df)} =======')
                    
    para_flag.train_sample_size = len(train_df)
    para_flag.valid_sample_size = len(valid_df)
        
    agent = Registed_Trainer(para_flag.trainer_name)(train_df, valid_df, DeepFace_Seq_Sys,  para_flag)
    agent.fit()
            

