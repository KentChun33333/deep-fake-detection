
import pandas as pd
import numpy as np

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
   