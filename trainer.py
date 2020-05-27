import pandas as pd 
import numpy as np 
import os 
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from registor import Registed_Trainer

# for special iterative trainer 

# for sub-dataset training 
# for lottery-theory 

@Registed_Trainer.regist
class IterativeDataTrainer():
    # seed data like 100 or 1000? 
    # Pseudo-labeling & cherry-picking unlabeled data
    # with dataset that is easily have leakage that hard for human to detect 
    
    # this is contractitionary to the pratice of how do you to outlier filtering.
    # 
    def __init__(self, 
                 train_df, 
                 valid_df, 
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
            model.build_dataset(self.train_df, self.valid_df, self.valid_df)    
            
            save_dir = f'hub_res_{hpara.model_version}'
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
                    # saves checkpoints to my_path whenever 'val_loss' has a new min
            checkpoint_callback = ModelCheckpoint(filepath=save_dir, save_top_k=3)

            # saves file like: /my/path/here/sample-mnist_epoch=02_val_loss=0.32.ckpt
            early_stop_callback = EarlyStopping(monitor='val_loss', 
                                                min_delta=0.02, 
                                                patience= hpara.early_stop_callback_patience, 
                                                verbose=False,
                                                mode='min')


            trainer = Trainer(gpus=[1,3], default_save_path=save_dir,
                              early_stop_callback=early_stop_callback,
                              checkpoint_callback = checkpoint_callback,
                              #distributed_backend='dp', 
                              accumulate_grad_batches=1, 
                              progress_bar_refresh_rate=1, log_save_interval=1,
                              check_val_every_n_epoch = 1, 
                              row_log_interval=1,
                              gradient_clip= 1., 
                              min_nb_epochs= 4,
                              max_nb_epochs= 100)     
            
            trainer.fit(model) 
            
            valid_df_pth = os.path.join(model.exp_save_path , f'validation_{model.current_epoch}.csv')
            try:
                self.check_error(valid_df_pth)
            except:
                pass 

    def check_error(self, df_pth):
        # it is kind of hard code and depend on the lightning module, 
        # 
        valid_df_file_pth_col = 'save_pth'
        
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

@Registed_Trainer.regist
class KfoldTrainer():
    # seed data like 100 or 1000? 
    # Pseudo-labeling & cherry-picking unlabeled data
    # with dataset that is easily have leakage that hard for human to detect 
    
    # this is contractitionary to the pratice of how do you to outlier filtering.
    # 
    def __init__(self, 
                 train_df, 
                 valid_df, 
                 lightning_module, 
                 hparams_of_module ):
        self.train_df, self.valid_df = train_df, valid_df 
        self.train_df.index = range(len(self.train_df))
        self.valid_df.index = range(len(self.valid_df))
        
        self.lightning_module = lightning_module
        
        self.module_hparams = hparams_of_module
        if not hasattr(lightning_module, "build_dataset"):
            raise Exception('need to have build_dataset in lightning_module to enable IterativeDataTrainer')
            
            
    def kfold_split(self, df):
        # 0~10, 10~20, 20~30, 30~40
        for ind, i in enumerate(range(4)):
            start = i*10
            end   = start+10
            temp_df = df[df.chunk>=start]
            temp_df = temp_df[temp_df.chunk<=end]
            yield df, ind

    def fit(self):
        hpara = self.module_hparams
        
        for new_train_df, kf_ind in self.kfold_split(self.train_df):
            hpara.kf_ind = kf_ind
            hpara.train_sample_size = len(new_train_df)
            hpara.valid_sample_size = len(self.valid_df)
            
            print(f'|========= Start KfoldTrainer - {kf_ind} =========|')
            print(hpara.__dict__)
            model = self.lightning_module(hpara) 

            # need to be relaxed 
            model.build_dataset(new_train_df, self.valid_df, self.valid_df)    
            
            save_dir = f'hub_res_{hpara.model_version}'
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
                    # saves checkpoints to my_path whenever 'val_loss' has a new min
            checkpoint_callback = ModelCheckpoint(filepath=save_dir, save_top_k=3)

            # saves file like: /my/path/here/sample-mnist_epoch=02_val_loss=0.32.ckpt
            early_stop_callback = EarlyStopping(monitor='val_loss', 
                                                min_delta=0.02, 
                                                patience= hpara.early_stop_callback_patience, 
                                                verbose=False,
                                                mode='min')

            trainer = Trainer(gpus=[1,3], default_save_path=save_dir,
                              early_stop_callback=early_stop_callback,
                              checkpoint_callback = checkpoint_callback,
                              #distributed_backend='dp', 
                              accumulate_grad_batches=1, 
                              progress_bar_refresh_rate=1, log_save_interval=1,
                              check_val_every_n_epoch = 1, 
                              row_log_interval=1,
                              gradient_clip= 1., 
                              min_nb_epochs= 4,
                              max_nb_epochs= 100)     
            
            trainer.fit(model) 
            
            valid_df_pth = os.path.join(model.exp_save_path , f'validation_{model.current_epoch}.csv')
