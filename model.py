
from torch import Tensor
from torch import nn
import numpy as np
import pandas as pd

from efficientnet_pytorch import EfficientNet
from collections import defaultdict

from tqdm import tqdm

import torch, os, kornia, math
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_sequence


from torch.autograd import Variable
import torchvision.models

class PretrainedModel(nn.Module):

    ERR_TRUNC_MSG = ("{} currently only supports to be truncated "
                     "by its last {} FC layer(s). Please choose a value "
                     "between 0 and {}.")

    ERR_MODEL = "{} is currently not supported as a pretrained model."

    SUPPORTED_MODEL_NAMES = ['resnet18', 'resnet34', 'resnet50',
                             'resnet101', 'resnet152',
                             'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn',
                             'vgg16', 'vgg16_bn', 'vgg19_bn', 'vgg19']
                             #'inception_v3']

    def __init__(self, model_name, layers_to_truncate=1):
        """Load the pretrained ResNet-152 and replace top fc layer."""
        super(PretrainedModel, self).__init__()

        if model_name not in self.SUPPORTED_MODEL_NAMES:
            raise NotImplementedError(self.ERR_MODEL.format(model_name))

        """
        if model_name == 'inception_v3':
            self.input_size = (3, 299, 299)
            layer_size = 1
            max_trunc = 1
        """
        if model_name.startswith('resnet'):
            self.input_size = (3, 224, 224)
            layer_size = 1
            max_trunc = 1
        elif model_name.startswith('vgg'):
            self.input_size = (3, 224, 224)
            layer_size = 3
            max_trunc = 3
        else:
            raise NotImplementedError(self.ERR_MODEL.format(model_name))

        if layers_to_truncate > max_trunc:
            raise ValueError(self.ERR_TRUNC_MSG.format(model_name, max_trunc, max_trunc))

        model = getattr(torchvision.models, model_name)(pretrained=True)

        if layers_to_truncate < 1:
            # Do not truncate
            self.pretrained_model = model
        else:
            # Truncate last FC layer(s)
            if model_name.startswith('vgg'):
                layers = list(model.classifier.children())
            else:
                layers = list(model.children())
            trunc = self._get_num_truncated_layers(layers_to_truncate, layer_size)
            last_layer = layers[trunc]

            # Delete the last layer(s).
            modules = layers[:trunc]
            if model_name.startswith('vgg'):
                self.pretrained_model = model
                self.pretrained_model.classifier = nn.Sequential(*modules)
            else:
                self.pretrained_model = nn.Sequential(*modules)

        # Freeze all parameters of pretrained model
        for param in self.pretrained_model.parameters():
                param.requires_grad = True

        # Switch model to eval mode (affects Dropout & BatchNorm)
        self.pretrained_model.eval()
        # TODO: Test if last_layer.in_features can be reliably used instead
        self.output_size = self._get_output_size()

    def _get_output_size(self):
        dummy_input = Variable(torch.rand(1, *self.input_size))
        output = self(dummy_input)
        output_size = output.data.view(-1).size(0)
        return output_size

    def _get_num_truncated_layers(self, num_to_trunc, layer_size, initial_layer_size=1):
        num = 0
        if num_to_trunc > 0:
            num += initial_layer_size
            num_to_trunc -= 1
        while num_to_trunc > 0:
            num += layer_size
            num_to_trunc -= 1
        return -num

    def forward(self, images):
        """Extract the image feature vectors."""
        features = self.pretrained_model(images)
        features = Variable(features.data)
        features = features.view(features.size(0), -1)
        return features
  
class EFF_extrac(EfficientNet):
    output_size = 62720
    #output_size = 68992  # B2 220 
    #output_size = 316800 # 480 B2 
    #output_size = 288000 # 480 B0 
    
    def __call__(self, x):
        return self.extract_features(x)
        
class SEBlock(nn.Module):
    def __init__(self, in_ch, r=8):
        super(SEBlock, self).__init__()

        self.linear_1 = nn.Linear(in_ch, in_ch//r)
        self.linear_2 = nn.Linear(in_ch//r, in_ch)
        self.init_weights()
        
    def init_weights(self):
        self.linear_1.weight.data.uniform_(-0.1, 0.1)
        self.linear_1.bias.data.fill_(0)
        self.linear_2.weight.data.uniform_(-0.1, 0.1)
        self.linear_2.bias.data.fill_(0)
        
    def forward(self, x):
        input_x = x

        x = F.relu(self.linear_1(x), inplace=True)
        x = self.linear_2(x)
        x = F.sigmoid(x)

        x = input_x * x

        return x

class TwoLSTMs(nn.Module):
    def __init__(self):
        self.rnn_one = nn.LSTM(input_size=10, hidden_size=20, num_layers=2)
        self.rnn_two = nn.LSTM(input_size=20, hidden_size=2)
    def forward(self, inp, h0, c0):
        output, (hn, cn) = rnn(inp, (h0, c0))
        output2, _ = rnn_two(output)
        return output2

class CNN_Only(nn.Module):
    def __init__(self, cnn_model_name ='resnet18', lstm1_output_size = 32, fc1_hidden_size=64, 
                 output_size=1, **kwargs):
        super().__init__()
        
        if cnn_model_name == 'Eff':
            
            self.cnn_model = EFF_extrac.from_name('efficientnet-b0')
            state_dict = torch.load('/test/torch_pretrain/efficientnet-b0-355c32eb.pth', map_location='cpu')
            self.cnn_model.load_state_dict(state_dict)
        else:
            self.cnn_model  = PretrainedModel(cnn_model_name)
            
        self.cnn_output_feature_sz = self.cnn_model.output_size
        
        self.fc1        = nn.Linear(self.cnn_output_feature_sz, fc1_hidden_size)
        self.seblock    = SEBlock(self.cnn_output_feature_sz)
        self.dropout    = nn.Dropout(0.2)        
        self.fc2        = nn.Linear(fc1_hidden_size, output_size)
        self.output_size = output_size
        self.init_weights()
    
    def init_weights(self):
        self.fc1.weight.data.uniform_(-0.1, 0.1)
        self.fc1.bias.data.fill_(0)
        self.fc2.weight.data.uniform_(-0.1, 0.1)
        self.fc2.bias.data.fill_(0)
        
    def forward(self, imgs:'Bacth:Frames:face:C:H:W'):
        B, frame, C, H, W = imgs.size()
        face = 1
        imgs = imgs[:, 0, :, :, :]
        
        imgs = self.dropout(self.cnn_model(imgs))
        imgs = self.seblock(imgs)
        imgs = self.fc1(imgs)
        y = F.sigmoid(self.fc2(imgs))#.view(B, -1)
        return y
    
class CNN_LSTM_Simple(nn.Module):
    def __init__(self, cnn_model_name ='resnet18', 
                 lstm1_output_size = 128 , fc1_hidden_size=64, 
                 output_size=1, hparam={}, **kwargs):
        super().__init__()
        
        if cnn_model_name == 'Eff':
            self.cnn_model = EFF_extrac.from_pretrained('efficientnet-b0')
        else:
            self.cnn_model  = PretrainedModel(cnn_model_name)
            
        self.cnn_output_feature_sz = self.cnn_model.output_size
                
        
        self.lstm1      = nn.LSTM(self.cnn_output_feature_sz, lstm1_output_size, 
                                  num_layers=hparam.lstm_layer, batch_first=True)

        self.fc1 = nn.Linear(lstm1_output_size, fc1_hidden_size)
        self.fc2 = nn.Linear(fc1_hidden_size, output_size)
        self.output_size = output_size

        
    def forward(self, imgs:'Bacth:Frames:C:H:W'):
        B, frame, C, H, W = imgs.size()
        # maybe put too much noise?
        
        x = imgs.reshape((B * frame, C, H, W))
        
        x = self.cnn_model(x)        

        x = x.view(B, frame, self.cnn_output_feature_sz)

        outputs, (ht, ct) = self.lstm1(x)
        
        y = (outputs[:, -1, :])
        y = F.elu(self.fc1(y))
        y = F.sigmoid(self.fc2(y))
        return y
    
class CNN_Only_Iter(nn.Module):
    def __init__(self, cnn_model_name ='resnet18', lstm1_output_size = 32, fc1_hidden_size=64, 
                 output_size=1, **kwargs):
        super().__init__()
        
        if cnn_model_name == 'Eff':
            self.cnn_model = EFF_extrac.from_pretrained('efficientnet-b0')
            dfs_freeze(self.cnn_model)
        else:
            self.cnn_model  = PretrainedModel(cnn_model_name)
            
        self.cnn_output_feature_sz = self.cnn_model.output_size
        
        lstm2_output_size = lstm1_output_size
        self.fc1        = nn.Linear(self.cnn_output_feature_sz, fc1_hidden_size)

        self.dropout    = nn.Dropout(0.2)
        self.seblock    = SEBlock(lstm2_output_size)
        self.fc2        = nn.Linear(fc1_hidden_size, output_size)
        self.output_size = output_size

        self.init_weights()
    
    def init_weights(self):
        self.fc1.weight.data.uniform_(-0.1, 0.1)
        self.fc1.bias.data.fill_(0)
        self.fc2.weight.data.uniform_(-0.1, 0.1)
        self.fc2.bias.data.fill_(0)
        
    def forward(self, imgs:'Bacth:Frames:face:C:H:W'):
        B, frame, face, C, H, W = imgs.size()
        #face = 1
        #imgs = imgs[:, :, 0, :, :, :]
        
        a = imgs.reshape((B *frame *face, C, H, W))
        imgs = self.cnn_model(a)
        a = imgs.reshape((B, frame *face, self.cnn_output_feature_sz))
        imgs = self.fc1(a)
        res_frame = []
        for i in range(frame *face):
            if self.output_size ==1:
                y = F.sigmoid(self.fc2(imgs[:, i, :]))#.view(B, -1)
            else:
                y = F.softmax(self.fc2(imgs[:, i, :]))#.view(B, -1)
            res_frame.append(y)
        y, ind = torch.max(torch.stack(res_frame), dim=0)
        return y
    
class CNN_LSTM(nn.Module):
    def __init__(self, cnn_model_name ='resnet18', 
                 lstm1_output_size = 32, fc1_hidden_size=2048, 
                 output_size=1, hparam={}, **kwargs):
        super().__init__()
        
        if cnn_model_name == 'Eff':
            self.cnn_model = EFF_extrac.from_name('efficientnet-b0')
            state_dict = torch.load('/test/torch_pretrain/efficientnet-b0-355c32eb.pth', map_location='cpu')
            #state_dict = torch.load('/test/torch_pretrain/efficientnet-b3-5fb5a3c3.pth', map_location='cpu')

            self.cnn_model.load_state_dict(state_dict)
            #dfs_freeze(self.cnn_model)
        else:
            self.cnn_model  = PretrainedModel(cnn_model_name)
            
        self.cnn_output_feature_sz = self.cnn_model.output_size
                
            
        self.fc1 = nn.Linear(self.cnn_output_feature_sz, fc1_hidden_size)
        #self.bn1 = nn.BatchNorm1d(fc1_hidden_size, momentum=0.01)
        #self.se_block1 = SEBlock(fc1_hidden_size, r=8)
        
        self.dropout    = nn.Dropout(hparam.dropout_rate)
        self.lstm1      = nn.LSTM(fc1_hidden_size, lstm1_output_size, 
                                  num_layers=hparam.lstm_layer, 
                                  batch_first=True)

        self.fc2 = nn.Linear(lstm1_output_size, hparam.fc2_hidden_size)
        #self.bn2 = nn.BatchNorm1d(fc2_hidden_size, momentum=0.01)
        self.fc3 = nn.Linear(hparam.fc2_hidden_size, output_size)
        self.output_size = output_size
        self.phase = 'train'
    #    self.transform = nn.Sequential(
    #kornia.color.AdjustBrightness(0.5),
    #kornia.color.AdjustContrast(0.7),
    #kornia.augmentation.RandomHorizontalFlip(0.5),
    #kornia.augmentation.RandomCrop((215, 215)),
     #   )
        self.init_weights()
    
    def init_weights(self):
        self.fc1.weight.data.uniform_(-0.1, 0.1)
        self.fc1.bias.data.fill_(0)
        
        self.fc2.weight.data.uniform_(-0.1, 0.1)
        self.fc2.bias.data.fill_(0)
        
        self.fc3.weight.data.uniform_(-0.1, 0.1)
        self.fc3.bias.data.fill_(0)
        
    def forward(self, imgs:'Bacth:Frames:face:C:H:W'):
        B, frame, C, H, W = imgs.size()
        # maybe put too much noise?
        x = imgs.reshape((B *frame , C, H, W))
        #if self.phase =='train':
        #    x = self.transform(x)

        x = F.interpolate(x, 240)

        x = self.cnn_model(x)        
        # print(x.size())
        # 12, 1480 7 7 
        x = x.view(B, frame , self.cnn_output_feature_sz)

        x = F.elu(self.fc1(x))
        #x = self.se_block1(x)
        outputs, (ht, ct) = self.lstm1(x)
        
        y = (outputs[:, -1, :])
        y = F.elu(self.fc2(y))
        y = F.sigmoid(self.fc3(y))
        return y
      
def dfs_freeze(model):
    for name, child in model.named_children():
        for param in child.parameters():
            param.requires_grad = False
        dfs_freeze(child)
    model.eval()