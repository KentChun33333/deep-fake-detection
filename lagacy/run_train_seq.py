import os
import pandas as pd 
import numpy as np 
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping
from argparse import Namespace

import json
from sklearn.model_selection import StratifiedShuffleSplit, GroupShuffleSplit
from sklearn.model_selection import GroupShuffleSplit
from copy import deepcopy
from ka_dataset import Ka2020DeepFackSeq 
from agent_seq import DeepFace_Seq_Sys
from datetime import datetime 

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

def balance_target_df(df, col):
    res= []
    max_value = df[col].value_counts().max()
    for v in list(set(df[col])):
        temp_df = df[df[col]==v]
        res.append(temp_df.sample(n=max_value, replace=True))
    return pd.concat(res, ignore_index=True)

def get_config_03():
    para_flag = Namespace()
    para_flag.batch_size = 8
    para_flag.learning_rate= 1e-4
    para_flag.fc1_hidden_size= 128*5
    
    para_flag.lstm1_output_size = 128
    para_flag.output_size=2
    para_flag.model_name = 'CNN_LSTM_3'
    para_flag.cnn_model_name = 'Eff'
    para_flag.seq_train_range = (5, 10)
    return para_flag

def get_config_03_iter():
    para_flag = Namespace()
    para_flag.batch_size = 12
    para_flag.learning_rate= 1e-4
    para_flag.fc1_hidden_size= 224
    
    para_flag.lstm1_output_size = 36
    para_flag.output_size=2
    para_flag.model_name = 'CNN_LSTM_3_iter'
    para_flag.cnn_model_name = 'Eff'
    para_flag.seq_train_range = (5, 10)
    return para_flag

def get_config_02():
    para_flag = Namespace()
    para_flag.batch_size = 12
    para_flag.learning_rate= 1e-4
    para_flag.fc1_hidden_size= 224
    
    para_flag.lstm1_output_size = 64
    para_flag.output_size=2
    para_flag.model_name = 'CNN_LSTM_2'
    para_flag.cnn_model_name = 'Eff'
    para_flag.seq_train_range = (5, 10)

    return para_flag

def get_config_01():
    para_flag = Namespace()
    para_flag.batch_size = 12
    para_flag.learning_rate= 1e-4
    para_flag.fc1_hidden_size = 224
    
    para_flag.lstm1_output_size = 64
    para_flag.output_size=2
    para_flag.model_name = 'CNN_LSTM'
    para_flag.cnn_model_name = 'Eff' # one key ... efficeinet so good...
    para_flag.seq_train_range = (5, 10)

    return para_flag


def get_config_00():
    para_flag = Namespace()
    para_flag.batch_size = 12
    para_flag.learning_rate = 0.004
    para_flag.fc1_hidden_size = 456
    
    para_flag.lstm1_output_size = 64
    para_flag.output_size=2
    para_flag.model_name = 'CNN_LSTM'
    para_flag.cnn_model_name = 'Eff'
    para_flag.seq_train_range = (5, 10)

    return para_flag

def get_config_04():
    # not that promising -> might need smaller 3m cus model or larger pretrain model
    para_flag = Namespace()
    para_flag.batch_size = 12
    para_flag.learning_rate= 0.0001
    para_flag.fc1_hidden_size= 2048
    para_flag.lstm1_output_size = 128
    para_flag.fc2_hidden_size = 64
    
    para_flag.output_size= 1
    para_flag.model_name = 'CNN_LSTM_4'
    para_flag.cnn_model_name = 'Meso' # no use
    para_flag.seq_min = 3
    para_flag.seq_max = 4
    para_flag.early_stop_callback_patience = 4
    para_flag.model_version = str(datetime.now()).split('.')[0].replace(':', '-').replace(' ','_')
    para_flag.train_mode = 'train'
    para_flag.preprocess_mode = "seq_seq_12_gap_5_full_480"
    para_flag.dataset_name = "Ka2020DeeFackFrameSeq"
    return para_flag

def get_config_05():
    # not that promising -> might need smaller 3m cus model or larger pretrain model
    # seq model need larger batch ? 
    para_flag = Namespace()
    para_flag.batch_size = 12
    para_flag.learning_rate= 0.0001
    para_flag.fc1_hidden_size= None

    para_flag.lstm1_output_size = 64
    para_flag.output_size= 2
    para_flag.model_name = 'CNN_LSTM_5'
    para_flag.cnn_model_name = 'Eff'
    para_flag.seq_min = 3
    para_flag.seq_max = 4
    para_flag.model_version = str(datetime.now()).split('.')[0].replace(':', '-').replace(' ','_')
    para_flag.preprocess_mode = "seq_seq_12_gap_5_align_face"
    para_flag.dataset_name = "Ka2020DeepFackSeq"
    return para_flag

def gen_get_config_01():
    # if the net too big --> shit, hard to train, overfitting 
    # keep net small and test on small pairs, 
    # 
    para_flag = Namespace()
    para_flag.batch_size = 12
    para_flag.learning_rate= 0.0004
    para_flag.output_size= 1
    para_flag.model_name = 'CNN_LSTM'
    para_flag.cnn_model_name = 'Eff' # one key ... efficeinet so good...
    para_flag.seq_min = 5
    para_flag.seq_max = 6
    # early stop para
    para_flag.early_stop_callback_patience = 5
    para_flag.fc1_hidden_size = 64
    para_flag.lstm1_output_size = 64
    para_flag.fc2_hidden_size = 32
    para_flag.dropout_rate = 0.1
    para_flag.lstm_layer = 1
    para_flag.model_version = str(datetime.now()).split('.')[0].replace(':', '-').replace(' ','_')
    para_flag.train_mode = 'train'
    para_flag.preprocess_mode = "seq_seq_12_gap_5_align_face"
    para_flag.dataset_name = "Ka2020DeepFackSeq"

    return para_flag

def gen_get_config_01_simple():
    para_flag = Namespace()
    para_flag.batch_size = 12
    para_flag.learning_rate= 0.0002
    para_flag.output_size= 1
    para_flag.model_name = 'CNN_LSTM_Simple'
    para_flag.cnn_model_name = 'Eff' # one key ... efficeinet so good...
    para_flag.seq_min = 5
    para_flag.seq_max = 6
    # early stop para
    para_flag.early_stop_callback_patience = 5
    para_flag.fc1_hidden_size = 64
    para_flag.lstm1_output_size = 128
#     para_flag.fc2_hidden_size = 64
    para_flag.dropout_rate = 0.1
    para_flag.lstm_layer = 1
    para_flag.model_version = str(datetime.now()).split('.')[0].replace(':', '-').replace(' ','_')
    para_flag.train_mode = 'train-raw'
    para_flag.preprocess_mode = "seq_seq_12_gap_5_align_face"
    para_flag.dataset_name = "Ka2020DeeFackSeq"
    return para_flag

#            "Ka2020DeepFackSeq":Ka2020DeepFackSeq, 

def get_config_CNN_Only_Iter():
    para_flag = Namespace()
    para_flag.batch_size = 12
    para_flag.learning_rate= 0.00001
    para_flag.fc1_hidden_size = 456
    para_flag.fc2_hidden_size = None
    para_flag.lstm1_output_size = 64
    para_flag.output_size=1
    para_flag.model_name = 'CNN_Only_Iter'
    para_flag.cnn_model_name = 'Eff'
    para_flag.seq_min = 8
    para_flag.seq_max = 9
    para_flag.model_version = str(datetime.now()).split('.')[0].replace(':', '-').replace(' ','_')
    para_flag.train_mode = 'train-raw'
    return para_flag  

def get_config_CNN_Only():
    para_flag = Namespace()
    para_flag.batch_size = 12
    para_flag.learning_rate= 0.0001
    para_flag.fc1_hidden_size = 444
    para_flag.fc2_hidden_size = None

    para_flag.lstm1_output_size = 64
    para_flag.output_size= 1
    para_flag.model_name = 'CNN_Only'
    para_flag.cnn_model_name = 'Eff'
    para_flag.seq_min = 3
    para_flag.seq_max = 9
    para_flag.model_version = str(datetime.now()).split('.')[0].replace(':', '-').replace(' ','_')
    para_flag.train_mode = 'train-raw'
    
    return para_flag

def select_with_file(df, col):
    ind = df[col].apply(lambda x: os.path.exists(x) ) 
    df = df[ind]
    return df 
       
            
class IterativeDataTrainer():
    # seed data like 100 or 1000? 
    # Pseudo-labeling & cherry-picking unlabeled data
    # with dataset that is easily have leakage that hard for human to detect 
    
    # this is contractitionary to the pratice of how do you to outlier filtering.
    # 
    def __init__(self, train_df, valid_df, 
                 lightning_module, 
                 hparams_of_module, 
                 max_ep = 2, 
                 error_rank_threshold =0.2):
        self.train_df, self.valid_df = train_df, valid_df 
        self.train_df.index = range(len(self.train_df))
        self.valid_df.index = range(len(self.valid_df))
        
        self.lightning_module = lightning_module
        
        self.module_hparams = hparams_of_module
        if not hasattr(lightning_module, "build_dataset"):
            raise Exception('need to have build_dataset in lightning_module to enable IterativeDataTrainer')
            
        self.module_hparams.IterativeDataTrainer_ep = 0 
        self.module_hparams.IterativeDataTrainer_error_threshold = error_rank_threshold  # mean top k error sample would add to training every time 
        self.module_hparams.IterativeDataTrainer_max_ep = max_ep
            
            
    def fit(self):
        hpara = self.module_hparams
        
        hpara.train_sample_size = len(self.train_df)
        hpara.valid_sample_size = len(self.valid_df)
        
        while hpara.IterativeDataTrainer_ep < hpara.IterativeDataTrainer_max_ep:
            hpara.IterativeDataTrainer_ep +=1
            print(f'|========= Start IterativeDataTrainer - {hpara.IterativeDataTrainer_ep} =========|')
            print(hpara.__dict__)
            model = self.lightning_module(hpara) 

            # need to be relaxed 
            model.build_dataset(self.train_df, self.valid_df, self.valid_df, hpara.train_mode)    
            
            early_stop_callback = EarlyStopping(monitor='val_loss', 
                                                min_delta=0.02, 
                                                patience= hpara.early_stop_callback_patience, 
                                                verbose=False,
                                                mode='min')

            trainer = Trainer(gpus=[1,2], default_save_path=f'hub_res_{hpara.model_version}',
                              early_stop_callback=early_stop_callback,
                              distributed_backend='dp', 
                              accumulate_grad_batches=1, progress_bar_refresh_rate=1, log_save_interval=1,
                              row_log_interval=1,
                              gradient_clip= 1., 
                              min_nb_epochs= 4,
                              max_nb_epochs= 100)     
            
            trainer.fit(model) 
            
            valid_df_pth = os.path.join(model.exp_save_path , f'validation_{model.current_epoch}.csv')
            
            self.check_error(valid_df_pth)

    def check_error(self, df_pth):
        # it is kind of hard code and depend on the lightning module, 
        # 
        valid_df_file_pth_col = 'save_10_pth'
        
        print('==== Check Error Samples ====')
        ddff = pd.read_csv(df_pth)
        
        ddff['P:Fake'] = ddff['prediction_result'].apply(
            lambda x: float(x.split('[')[1].split(']')[0]))

        del  ddff['prediction_result']
        ddff['target'] = ddff.target.apply(
            lambda x:float(x.split(',')[0].replace('tensor(','')))
    
        ddff['ind_loss'] = abs(ddff['target']- ddff['P:Fake'])
        print(f'Error {len(ddff[ddff.ind_loss>0.5])} with Right {len(ddff[ddff.ind_loss<=0.5])}')
        err_df = ddff[ddff["ind_loss"]>0.5].sort_values("ind_loss", ascending=False)
        
        # top % error sample 
        err_df = err_df[:int(self.module_hparams.IterativeDataTrainer_error_threshold*len(err_df))]
    
        print(err_df.target.value_counts())
        
        obj = err_df.file_name.values
        
        err_df = self.valid_df[self.valid_df[valid_df_file_pth_col].isin(obj)]
        new_val_df = self.valid_df[~self.valid_df[valid_df_file_pth_col].isin(obj)]
        
        self.train_df = pd.concat([self.train_df, err_df], ignore_index=True)

        self.valid_df = new_val_df
        self.train_df.index = range(len(self.train_df))
        self.valid_df.index = range(len(self.valid_df))
        self.train_df = balance_target_df(self.train_df, 'target')

        
def prepare_train_valid(df, preprocess_mode='_seq_12_gap_5_full_480'):
    #df['save_10_pth'] = df['save_10_pth'].apply(lambda x: x.replace('multi', 'seq_seq_12_gap_5_align_face'))
    df['save_10_pth'] = df['save_10_pth'].apply(lambda x: x.replace('multi', preprocess_mode))

    
    df = select_with_file(df, 'save_10_pth')
    df['target'] = df['target'].apply(lambda x: 1 if x=='FAKE' else 0)    
    
    train_df = df[df['chunk']<40]
    valid_df = df[df['chunk']>=40]
    if len(valid_df)==0:
        thrsh = int(len(train_df)*0.8)
        train_df, valid_df = train_df[:thrsh], train_df[thrsh: ]
    #train_df, test_df = get_GroupShuffleSplit(train_df)    
    for i in [train_df, valid_df]:
        i.index = range(len(i))
    
    print(len(train_df), len(valid_df),)
        
    #train_df = balance_hook_df(train_df, 'target')
    valid_df.index = range(len(valid_df))
    #test_df.index = range(len(test_df))    
    return train_df, valid_df
    
if 1==1:
    
    # data leakage problem 
    df = pd.read_csv('original_below_30.csv')
    #

    eps_list = [5, 5, 5]
    
    for para_flag in [
        #get_config_04(), 
        gen_get_config_01(),    
        #gen_get_config_01_simple(), 
            
        ]:
        
        train_df, valid_df = prepare_train_valid(df, para_flag.preprocess_mode)
        
        for ind, train_size_clip in enumerate([3000  ]):
            
            print(f'====== Start Training {para_flag.model_name} with Sample {train_size_clip} =======')
            temp_train_df = balance_target_df(train_df[:train_size_clip], 'target')
            #valid_df = pd.concat([valid_df, train_df[train_size_clip:]])
            print(temp_train_df.target.value_counts())
            
            para_flag.train_sample_size = len(temp_train_df)
            para_flag.valid_sample_size = len(valid_df)
            
            agent = IterativeDataTrainer(temp_train_df, valid_df, DeepFace_Seq_Sys,  para_flag)
            agent.fit()
            

