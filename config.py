from argparse import Namespace
from datetime import datetime 
from registor import Registed_Config

@Registed_Config.regist
class BaseConfig():
    def __init__(self):
        self.para_flag = Namespace()
        self.trainer_info()
        self.dataset_info()
        self.model_info()
        self.task_info()

    def get_config(self):
        return self.para_flag

    def task_info(self):
        self.para_flag.model_version = str(datetime.now()).split('.')[0].replace(':', '-').replace(' ','_')

    def trainer_info(self):
        self.para_flag.trainer_name = 'KfoldTrainer'
        self.para_flag.early_stop_callback_patience = 7
        self.para_flag.batch_size = 8
        self.para_flag.learning_rate= 1e-6
    
    def model_info(self):
        self.para_flag.model_name = 'CNN_LSTM'
        self.para_flag.cnn_model_name = 'Eff' # one key ... efficeinet so good...
        self.para_flag.fc1_hidden_size = 128
        self.para_flag.lstm1_output_size = 128
        self.para_flag.fc2_hidden_size = 128
        self.para_flag.dropout_rate = 0.1
        self.para_flag.lstm_layer = 1
        self.para_flag.output_size= 1

    def dataset_info(self):
        self.para_flag.dataset_name = "Ka2020DeepFackSeq"
        self.para_flag.seq_max = 15
        self.para_flag.seq_min = 10

@Registed_Config.regist
class BSL_CNN_LSTM_Simple(BaseConfig):
    def __init__(self):
        super().__init__()
        
    def model_info(self):
        self.para_flag.model_name = 'CNN_LSTM_Simple'
        self.para_flag.cnn_model_name = 'Eff' # one key ... efficeinet so good...
        self.para_flag.lstm1_output_size = 128
        self.para_flag.fc1_hidden_size = 64
        self.para_flag.dropout_rate = 0.1
        self.para_flag.lstm_layer = 1
        self.para_flag.output_size= 1

@Registed_Config.regist
class BSL_CNN_LSTM_Simple2(BaseConfig):
    def __init__(self):
        super().__init__()
        self.para_flag.learning_rate= 3e-6

    def model_info(self):
        self.para_flag.model_name = 'CNN_LSTM_Simple'
        self.para_flag.cnn_model_name = 'Eff' # one key ... efficeinet so good...
        self.para_flag.lstm1_output_size = 128*4
        self.para_flag.fc1_hidden_size = 128
        self.para_flag.dropout_rate = 0.1
        self.para_flag.lstm_layer = 1
        self.para_flag.output_size= 1

@Registed_Config.regist
class BSL_CNN(BaseConfig):
    def __init__(self):
        super().__init__()
        self.para_flag.learning_rate= 1e-4
        self.para_flag.batch_size = 14

    def model_info(self):
        self.para_flag.model_name = 'CNN_Only'
        self.para_flag.cnn_model_name = 'Eff' # one key ... efficeinet so good...
        self.para_flag.lstm1_output_size = 128
        self.para_flag.fc1_hidden_size = 512
        self.para_flag.dropout_rate = 0.1
        self.para_flag.lstm_layer = 1
        self.para_flag.output_size= 1
