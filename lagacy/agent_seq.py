from ka_dataset import Ka2020DeepFackSeq, Ka2020DeeFackFrameSeq
from pytorch_lightning import Trainer
from argparse import Namespace

import pytorch_lightning as pl
from torch import Tensor
from torch import nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from efficientnet_pytorch import EfficientNet


from collections import defaultdict

from tqdm import tqdm

import torch, os, kornia
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
                param.requires_grad = False

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
    #output_size = 68992 # B2 220 
    #output_size = 316800 # 480 B2 
    #output_size = 288000# 480 B0 
    
    def __call__(self, x):
        return self.extract_features(x)
        
        
class Meso4(nn.Module):
    def __init__(self, num_classes=2):
        super(Meso4, self).__init__()
        self.num_classes = num_classes
        self.conv1 = nn.Conv2d(3, 8, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(8)
        self.relu = nn.ReLU(inplace=True)
        self.leakyrelu = nn.LeakyReLU(0.1)

        self.conv2 = nn.Conv2d(8, 8, 5, padding=2, bias=False)
        self.bn2 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(8, 16, 5, padding=2, bias=False)
        self.conv4 = nn.Conv2d(16, 16, 5, padding=2, bias=False)
        self.maxpooling1 = nn.AvgPool2d(kernel_size=(2, 2))
        self.maxpooling2 = nn.AvgPool2d(kernel_size=(4, 4))
        #flatten: x = x.view(x.size(0), -1)
#         self.dropout = nn.Dropout2d(0.5)
#         self.fc1 = nn.Linear(16*8*8, 16)
#         self.fc2 = nn.Linear(16, num_classes)

    def forward(self, input):
        x = self.conv1(input) #(8, 256, 256)
        x = self.relu(x)
        x = self.bn1(x)
        x = self.maxpooling1(x) #(8, 128, 128)

        x = self.conv2(x) #(8, 128, 128)
        x = self.relu(x)
        x = self.bn1(x)
        x = self.maxpooling1(x) #(8, 64, 64)

        x = self.conv3(x) #(16, 64, 64)
        x = self.relu(x)
        x = self.bn2(x)
        x = self.maxpooling1(x) #(16, 32, 32)

        x = self.conv4(x) #(16, 32, 32)
#         x = self.relu(x)
#         x = self.bn2(x)
#         x = self.maxpooling2(x) #(16, 8, 8)
#         x = x.view(x.size(0), -1) #(Batch, 16*8*8)
#         x = self.dropout(x)
#         x = self.fc1(x) #(Batch, 16)
#         x = self.leakyrelu(x)
#         x = self.dropout(x)
#         x = self.fc2(x)

        return x

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
        face = 1
        imgs = imgs[:, :, 0, :, :, :]
        
        a = imgs.reshape((B *frame *face, C, H, W))
        imgs = self.dropout(self.cnn_model(a))
        a = imgs.reshape((B, frame *face, self.cnn_output_feature_sz))
        imgs = self.fc1(a)
        if self.output_size ==1:
            y = F.sigmoid(self.fc2(imgs[:, -1, :]))#.view(B, -1)
        else:
            y = F.softmax(self.fc2(imgs[:, -1, :]))#.view(B, -1)
        return y
    
class CNN_Only_Iter(nn.Module):
    """
    check the effect of fc1 after B, Seq, Img-feature, wether it would lead to a batch mix-up
    since CNN_Only could converge, and dont have the issue of the output of lstm would gradually become the same in one batch 
    
    turns out it works like a charm...with face =1 , 
    
    """
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
            self.cnn_model = EFF_extrac.from_pretrained('efficientnet-b0')
            dfs_freeze(self.cnn_model)
        else:
            self.cnn_model  = PretrainedModel(cnn_model_name)
        self.cnn_output_feature_sz = self.cnn_model.output_size
                
            
        self.fc1 = nn.Linear(self.cnn_output_feature_sz, fc1_hidden_size)
        #self.bn1 = nn.BatchNorm1d(fc1_hidden_size, momentum=0.01)
        #self.se_block1 = SEBlock(fc1_hidden_size, r=8)
        
        self.dropout    = nn.Dropout(hparam.dropout_rate)
        self.lstm1      = nn.LSTM(fc1_hidden_size, lstm1_output_size, 
                                  num_layers=hparam.lstm_layer, batch_first=True)

        self.fc2 = nn.Linear(lstm1_output_size, hparam.fc2_hidden_size)
        #self.bn2 = nn.BatchNorm1d(fc2_hidden_size, momentum=0.01)
        self.fc3 = nn.Linear(hparam.fc2_hidden_size, output_size)
        self.output_size = output_size
        self.phase = 'train'
        self.transform = nn.Sequential(
    kornia.color.AdjustBrightness(0.5),
    kornia.color.AdjustGamma(gamma=2.),
    kornia.color.AdjustContrast(0.7),
    kornia.augmentation.RandomHorizontalFlip(0.5),
    #kornia.augmentation.RandomCrop((215, 215)),
        )
        self.init_weights()
    
    def init_weights(self):
        self.fc1.weight.data.uniform_(-0.1, 0.1)
        self.fc1.bias.data.fill_(0)
        
        self.fc2.weight.data.uniform_(-0.1, 0.1)
        self.fc2.bias.data.fill_(0)
        
        self.fc3.weight.data.uniform_(-0.1, 0.1)
        self.fc3.bias.data.fill_(0)
        
    def forward(self, imgs:'Bacth:Frames:face:C:H:W'):
        B, frame, face, C, H, W = imgs.size()
        # maybe put too much noise?
        x = imgs.reshape((B *frame *face, C, H, W))
        if self.phase =='train':
            x = self.transform(x)
        
        x = F.interpolate(x, 224)
        x = self.cnn_model(x)        
        # print(x.size())
        # 12, 1480 7 7 
        x = x.view(B, frame * face, self.cnn_output_feature_sz)

        x = F.elu(self.fc1(x))
        #x = self.se_block1(x)
        outputs, (ht, ct) = self.lstm1(x)
        
        y = (outputs[:, -1, :])
        y = F.relu(self.fc2(y))
        # print(outputs[:, -1, :]==ht[-1]) :: True
        if self.output_size ==1:
            y = F.sigmoid(self.fc3(y))
        else:
            y = F.softmax(self.fc3(y))#.view(B, -1)
        return y

class CNN_LSTM_Simple(nn.Module):
    def __init__(self, cnn_model_name ='resnet18', 
                 lstm1_output_size = 128 , fc1_hidden_size=64, 
                 output_size=1, hparam={}, **kwargs):
        super().__init__()
        
        if cnn_model_name == 'Eff':
            self.cnn_model = EFF_extrac.from_pretrained('efficientnet-b0')
            dfs_freeze(self.cnn_model)
        else:
            self.cnn_model  = PretrainedModel(cnn_model_name)
        self.cnn_output_feature_sz = self.cnn_model.output_size
                
        
        self.lstm1      = nn.LSTM(self.cnn_output_feature_sz, lstm1_output_size, 
                                  num_layers=hparam.lstm_layer, batch_first=True)
        # 128(lstm), 64, 1
        self.fc1 = nn.Linear(lstm1_output_size, fc1_hidden_size)
        self.fc2 = nn.Linear(fc1_hidden_size, output_size)
        self.output_size = output_size
        self.init_weights()
    
    def init_weights(self):
        self.fc1.weight.data.uniform_(-0.1, 0.1)
        self.fc1.bias.data.fill_(0)
        self.fc2.weight.data.uniform_(-0.1, 0.1)
        self.fc2.bias.data.fill_(0)
        
    def forward(self, imgs:'Bacth:Frames:face:C:H:W'):
        B, frame, face, C, H, W = imgs.size()
        # maybe put too much noise?
        x = imgs.reshape((B *frame *face, C, H, W))
        
        x = F.interpolate(x, 224)
        # korina augmentattion 

        x = self.cnn_model(x)        
        # print(x.size())
        # 12, 1480 7 7 
        x = x.view(B, frame * face, self.cnn_output_feature_sz)

        outputs, (ht, ct) = self.lstm1(x)
        
        y = (outputs[:, -1, :])
        y = F.elu(self.fc1(y))
        # print(outputs[:, -1, :]==ht[-1]) :: True
        y = F.sigmoid(self.fc2(y))
        return y
    
class CNN_LSTM_2(nn.Module):
    def __init__(self, cnn_model_name ='resnet101', lstm1_output_size = 32, fc1_hidden_size=64, 
                 output_size=1, **kwargs):
        super().__init__()
        
        if cnn_model_name == 'Eff':
            self.cnn_model = EFF_extrac.from_pretrained('efficientnet-b0')
            dfs_freeze(self.cnn_model)
        else:
            self.cnn_model  = PretrainedModel(cnn_model_name)
            
        self.cnn_output_feature_sz = self.cnn_model.output_size

        self.fc1        = nn.Linear(self.cnn_output_feature_sz, fc1_hidden_size)
        self.lstm1      = nn.LSTM(fc1_hidden_size, lstm1_output_size, batch_first=True)
        #self.lstm2      = nn.LSTM(lstm1_output_size, lstm2_output_size, batch_first=True)
        self.dropout    = nn.Dropout(0.1)
        self.seblock    = SEBlock(lstm1_output_size)
        self.fc2        = nn.Linear(lstm1_output_size, output_size)
        self.init_weights()
    
    def init_weights(self):
        self.fc1.weight.data.uniform_(-0.1, 0.1)
        self.fc1.bias.data.fill_(0)
        self.fc2.weight.data.uniform_(-0.1, 0.1)
        self.fc2.bias.data.fill_(0)
        
    def forward(self, imgs:'Bacth:Frames:face:C:H:W'):
        B, n_frame, face, C, H, W = imgs.size()
        #face = 1
        #imgs = imgs[:, :, 0, :, :, :]
        a = imgs.reshape((B*n_frame*face, C, H, W))
        imgs = self.cnn_model(a)
        a = imgs.reshape((B, n_frame*face, self.cnn_output_feature_sz))
        imgs = self.fc1(a)
        hiddens, _ = self.lstm1(imgs)
        #hiddens = self.dropout(hiddens[:, 0, :])
        y = F.softmax(self.fc2(hiddens[:, -1, :]))#.view(B, -1)
        return y

class CNN_LSTM_3(nn.Module):
    def __init__(self, cnn_model_name ='resnet101', lstm1_output_size = 32, fc1_hidden_size=64, 
                 output_size=1, **kwargs):
        super().__init__()
        
        if cnn_model_name == 'Eff':
            self.cnn_model = EFF_extrac.from_pretrained('efficientnet-b0')
            dfs_freeze(self.cnn_model)
        else:
            self.cnn_model  = PretrainedModel(cnn_model_name)
            
        self.cnn_output_feature_sz = self.cnn_model.output_size

        self.seblock_1    = SEBlock(self.cnn_output_feature_sz)
        self.dropout    = nn.Dropout(0.1)
        self.fc1        = nn.Linear(self.cnn_output_feature_sz, fc1_hidden_size)
        
        self.lstm1      = nn.LSTM(fc1_hidden_size, lstm1_output_size, batch_first=True)
        #self.lstm2      = nn.LSTM(lstm1_output_size, lstm2_output_size, batch_first=True)
        
        self.seblock_2    = SEBlock(lstm1_output_size)
        self.fc2        = nn.Linear(lstm1_output_size, output_size)
        self.init_weights()
    
    def init_weights(self):
        self.fc1.weight.data.uniform_(-0.1, 0.1)
        self.fc1.bias.data.fill_(0)
        self.fc2.weight.data.uniform_(-0.1, 0.1)
        self.fc2.bias.data.fill_(0)
        
    def forward(self, imgs:'Bacth:Frames:face:C:H:W'):
        B, n_frame, face, C, H, W = imgs.size()

        x = imgs.reshape((B*n_frame*face, C, H, W))

        x = self.cnn_model(x)
        
        x = x.view(B, n_frame*face, self.cnn_output_feature_sz)
        x = self.fc1(self.dropout(self.seblock_1(x)))
        hiddens, _ = self.lstm1(x)
        x = hiddens[:, -1, :]
        y = F.softmax(self.fc2(x)).view(B, -1)
        return y

class CNN_LSTM_4(nn.Module):
    def __init__(self, lstm1_output_size = 32, 
                 fc1_hidden_size=64, 
                 output_size=1, hparam={}, **kwargs):
        super().__init__()
        
        self.cnn_model  = Meso4()
        self.cnn_output_feature_sz =14400 # 12544 # 57600# 12544

        self.seblock_1    = SEBlock(self.cnn_output_feature_sz)
        self.dropout    = nn.Dropout(0.2)
        self.fc1        = nn.Linear(self.cnn_output_feature_sz, fc1_hidden_size)
        self.lstm1      = nn.LSTM(fc1_hidden_size, lstm1_output_size, batch_first=True)
        #self.lstm2      = nn.LSTM(lstm1_output_size, lstm2_output_size, batch_first=True)
        
        #self.seblock_2    = SEBlock(lstm1_output_size)
        self.fc2        = nn.Linear(lstm1_output_size, hparam.fc2_hidden_size)
        self.fc3        = nn.Linear(hparam.fc2_hidden_size, output_size)

        self.init_weights()
    
    def init_weights(self):
        self.fc1.weight.data.uniform_(-0.1, 0.1)
        self.fc1.bias.data.fill_(0)
        self.fc2.weight.data.uniform_(-0.1, 0.1)
        self.fc2.bias.data.fill_(0)
        self.fc3.weight.data.uniform_(-0.1, 0.1)
        self.fc3.bias.data.fill_(0)
        
    def forward(self, imgs:'Bacth:Frames:face:C:H:W'):
        B, n_frame, face, C, H, W = imgs.size()

        x = imgs.reshape((B*n_frame*face, C, H, W))
        x = self.cnn_model(x)
        
        x = x.view(B, n_frame*face, self.cnn_output_feature_sz)
        
        x = self.fc1(self.seblock_1(self.dropout(x)))
        
        hiddens, _ = self.lstm1(x)
        x = hiddens[:, -1, :]
        y = F.elu(self.fc2(x))#.view(B, -1)
        y = F.sigmoid(self.fc3(y))#.view(B, -1)
        return y

class CNN_LSTM_5(nn.Module):
    def __init__(self, cnn_model_name ='resnet101', lstm1_output_size = 32, 
                 fc1_hidden_size=64, 
                 
                 output_size=1, **kwargs):
        super().__init__()
        
        if cnn_model_name == 'Eff':
            self.cnn_model = EFF_extrac.from_pretrained('efficientnet-b0')
            dfs_freeze(self.cnn_model)
        else:
            self.cnn_model  = PretrainedModel(cnn_model_name)
        self.cnn_output_feature_sz = self.cnn_model.output_size

        self.seblock_1    = SEBlock(self.cnn_output_feature_sz)
        self.dropout    = nn.Dropout(0.1)
        
        self.lstm1      = nn.LSTM(self.cnn_output_feature_sz, lstm1_output_size, batch_first=True)
        #self.lstm2      = nn.LSTM(lstm1_output_size, lstm2_output_size, batch_first=True)
        
        self.seblock_2    = SEBlock(lstm1_output_size)
        self.fc2        = nn.Linear(lstm1_output_size, output_size)
        self.init_weights()
    
    def init_weights(self):
        self.fc2.weight.data.uniform_(-0.1, 0.1)
        self.fc2.bias.data.fill_(0)
        
    def forward(self, imgs:'Bacth:Frames:face:C:H:W'):
        B, F, face, C, H, W = imgs.size()

        x = imgs.reshape((B*F*face, C, H, W))
        x = self.cnn_model(x)
        
        x = x.reshape((B, F*face, self.cnn_output_feature_sz))
        x = self.dropout(self.seblock_1(x))
        hiddens, _ = self.lstm1(x)
        x = self.seblock_2(hiddens[:, -1, :])
        y = F.softmax(self.fc2(x)).view(B, -1)
        return y

class CNN_LSTM_3_iter(nn.Module):
    def __init__(self, cnn_model_name ='resnet101', lstm1_output_size = 32, fc1_hidden_size=64, 
                 output_size=1, **kwargs):
        super().__init__()
        
        if cnn_model_name == 'Eff':
            self.cnn_model = EFF_extrac.from_pretrained('efficientnet-b0')
            dfs_freeze(self.cnn_model)
        else:
            self.cnn_model  = PretrainedModel(cnn_model_name)
        self.cnn_output_feature_sz = self.cnn_model.output_size

        self.seblock_1    = SEBlock(self.cnn_output_feature_sz)
        self.dropout    = nn.Dropout(0.1)
        self.fc1        = nn.Linear(self.cnn_output_feature_sz, fc1_hidden_size)
        self.layer_norm = nn.LayerNorm(fc1_hidden_size, eps=1e-12)
        self.lstm1      = nn.LSTM(fc1_hidden_size, lstm1_output_size, batch_first=True)
        #self.lstm2      = nn.LSTM(lstm1_output_size, lstm2_output_size, batch_first=True)
        
        self.seblock_2    = SEBlock(lstm1_output_size)
        self.fc2        = nn.Linear(lstm1_output_size, output_size)
        self.init_weights()
    
    def init_weights(self):
        self.fc1.weight.data.uniform_(-0.1, 0.1)
        self.fc1.bias.data.fill_(0)
        self.fc2.weight.data.uniform_(-0.1, 0.1)
        self.fc2.bias.data.fill_(0)
        
    def forward(self, imgs:'Bacth:Frames:face:C:H:W'):
        B, frame, face, C, H, W = imgs.size()
        res_frame = []
        frame=2
        for face_id in range(face)[:1]:
            x = imgs[:, :frame, face_id, :, : , :]
            res_temp =[] # frame-wise 
            for j in range(B):
                res_temp.append(self.cnn_model(x[j, :, :, :, :]))
            x = torch.stack(res_temp)
            # 
            x = self.fc1(x)
            # x = x.reshape((B*frame, C, H, W))
            # x = self.cnn_model(x)
            #x = x.reshape((B, frame, self.cnn_output_feature_sz))
            
            #x = self.layer_norm(x) # B*2*2048
            hiddens, _ = self.lstm1(x)
            
            
            
            x = self.seblock_2(hiddens[:, -1, :])
            y = F.softmax(self.fc2(x))#.view(B, -1)
            res_frame.append(y)
        res_frame, ind = torch.max(torch.stack(res_frame), dim=0)
        return y
    
    
    
class DeepFace_Seq_Sys(pl.LightningModule):

    def __init__(self, hparams):
        super().__init__()
        # train, test, train_labels, 
        self.hparams = hparams
        
        self.regist_model = {
            "CNN_LSTM_2": CNN_LSTM_2,
            "CNN_LSTM"  : CNN_LSTM, 
            "CNN_LSTM_3": CNN_LSTM_3,
            "CNN_LSTM_4": CNN_LSTM_4, 
            "CNN_LSTM_5": CNN_LSTM_5, 
            "CNN_LSTM_3_iter": CNN_LSTM_3_iter,
            "CNN_Only": CNN_Only,
            "CNN_Only_Iter": CNN_Only_Iter, 
            "CNN_LSTM_Simple":CNN_LSTM_Simple, 
        }
        
        self.regist_dataset = {
            "Ka2020DeepFackSeq":Ka2020DeepFackSeq, 
            "Ka2020DeeFackFrameSeq": Ka2020DeeFackFrameSeq, 
        }
        # change to 2 node 
        self.model = self.regist_model[hparams.model_name](
                              cnn_model_name = self.hparams.cnn_model_name,
                              lstm1_output_size = self.hparams.lstm1_output_size, 
                              fc1_hidden_size = self.hparams.fc1_hidden_size, 
                              output_size = self.hparams.output_size, 
            hparam = hparams)
        
        self.batch_size = hparams.batch_size
        self.learning_rate = hparams.learning_rate

        self.shuffle = True
        if self.hparams.output_size==1:
            self.criterion = nn.MSELoss()
            self.criterion = nn.BCEWithLogitsLoss()
            # print('MSE loss')
        else:
            self.criterion = nn.CrossEntropyLoss()
        self.validation_result = defaultdict(list)
        self.last_training_avg_loss = 0 # init

    def build_dataset(self, train_df, valid_df, test_df, train_model='train'):
            ## get only one fake and trian
        if train_model=='train':
            print('==== Balance Original ====')
            train_df_fake = train_df[train_df.target==1]
            train_df_real = train_df[train_df.target!=1]
            inter = set(train_df_fake.original).intersection(set(train_df_real.original))
            train_df_fake = select_by_original_(train_df_fake[train_df_fake['original'].isin(inter)])
            train_df_real = select_by_original_(train_df_real[train_df_real['original'].isin(inter)])    
            train_df = pd.concat([train_df_fake, train_df_real])
        
        self.train_dataset = self.regist_dataset[self.hparams.dataset_name](train_df, phase=train_model)
        self.vaild_dataset = self.regist_dataset[self.hparams.dataset_name](valid_df, phase='valid')
        self.test_dataset = self.regist_dataset[self.hparams.dataset_name](test_df, phase='test')

        
    def forward(self, batch):
        # chose only one face -w- shit 
        # [5, 10, 1, 224, 224, 3]
        #print(imgs['frames'])
        
        # Frame Selection should be in dataset..
        if self.hparams.dataset_name == 'Ka2020DeepFackSeq':
            x =  batch['frames']
            frame_seq_n = np.random.randint(self.hparams.seq_min, self.hparams.seq_max)
            frame_start = 0 
            
        else:
            x = batch['frames'].unsqueeze(2)
            frame_seq_n = 2 # 40
            frame_start = 0 # 20
        
        x = x.permute(0, 1, 2, 5, 4, 3)
        # 

        y_hat = self.model( x)
        return y_hat

    def get_loss(self, y_hat, y):
        #y_hat = torch.clamp(y_hat, min=0.3, max=0.7)

        loss = self.criterion(y_hat, y)
        return loss
    

    def training_step(self, batch, batch_idx):
        
        if self.hparams.output_size != 1: 
            y = batch['target'].long()
        else:
            y = batch['target'].float().unsqueeze(1)
        y_hat = self.forward(batch)
        
        predict = (y_hat > 0.5).float() * 1
        
        # mix loss 
        acc = (predict==y).float().mean() * 1
        
        # only update bad
        loss = self.get_loss(y_hat, y)/2 + (1-acc)/2
                
        log_dict = {'train_loss': loss, 'Tr-Acc': acc}
                    
        return {'loss': loss, 'log': log_dict}
    
    def validation_step(self, batch, batch_idx):

        if self.hparams.output_size != 1: 
            y = batch['target'].long()
        else:
            y = batch['target'].float().unsqueeze(1)
        y_hat = self.forward(batch)
        
        predict = (y_hat > 0.5).float() * 1
        
        # mix loss 
        acc = (predict==y).float().mean() * 1
        
        # only update bad
        loss = self.get_loss(y_hat, y)/2 + (1-acc)/2
        #loss = self.get_loss(y_hat, y) 
        # will see here -> size -> here -size, 
#         print('here ')
#         print(y_hat.size()[0])

#         # here because of doing multi-ple gpu, .... 
#         print('len of simaple-id in batch {}'.format(len(batch['file_name']))) 
#         print('batch target size {}'.format(batch['target'].size())) # if it is a tensor, it is good... 
#         print('batch frames size {}'.format(batch['batch_idx'].size()))
#         try:
#             print('batch original size {}'.format(batch['original'].size()))
#         except:
#             print('batch original size {}'.format(len(batch['original'])))
            

        for i in range(y_hat.size()[0]):
            self.validation_result['prediction_result'].append(y_hat[i])
            self.validation_result['file_name'].append(batch['file_name'][int(batch['batch_idx'][i].item())])
            self.validation_result['original'].append(batch['original'][i])
            self.validation_result['target'].append(batch['target'][i])
        return {'val_loss': loss, 'predict': predict.squeeze(), 'target': batch['target'].float() }
    
    def test_step(self, batch, batch_idx):
        
        if self.hparams.output_size != 1: 
            y = batch['target'].long()
        else:
            y = batch['target'].float().unsqueeze(1)
        y_hat = self.forward(batch)
        loss = self.get_loss(y_hat, y)

        return {'val_loss': loss }
    
    # training step end is doing within batch for parrellel training 
    def training_end(self, outputs):
        self.model.phase='train'
        # this out is now the full size of the batch
        try:
            avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        except:
            avg_loss = outputs['loss']
        self.last_training_avg_loss = avg_loss
        # this softmax now uses the full batch size
        #loss = nce_loss(loss)
        return {'loss': avg_loss, }
    
    def validation_epoch_end(self, outputs):
        self.model.phase='valid'

        avg_loss = torch.cat([x['val_loss'] for x in outputs]).mean()
        predict = torch.cat([x['predict'] for x in outputs])
        target = torch.cat([x['target'] for x in outputs])
        acc = (predict==target).float().mean() * 1
        
        log_dict = {'val_loss': avg_loss, 
                    'train_loss': self.last_training_avg_loss, 
                    'ep': self.current_epoch, 
                    'Acc': acc, 
                   }
        
        log_dict.update(self.hparams.__dict__)
        log_df = pd.DataFrame()
        log_df = log_df.append(log_dict, ignore_index=True)
        
        if not self.exp_save_path:
            self.exp_save_path = self.trainer.default_save_path
            
        path = os.path.join(self.exp_save_path , 'log.csv')
        
        if os.path.exists(path):
            log_df.to_csv(path , mode='a', header=False )
        else:
            log_df.to_csv(path)
        
        df = pd.DataFrame(self.validation_result)

        df.to_csv(os.path.join(self.exp_save_path , f'validation_{self.current_epoch}.csv'))
        self.validation_result = defaultdict(list)
        print('SAVE : ', os.path.join(self.exp_save_path , f'validation_{self.current_epoch}.csv'))
        
        return { 'val_loss': avg_loss, 'progress_bar':{'val_loss': avg_loss, 'Acc': acc}, }
        
    def configure_optimizers(self):
        # REQUIRED
        #opt_mizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        opt_mizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate, )
        
        #opt_mizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=0.9)
        #scheduler = torch.optim.lr_scheduler.CyclicLR(opt_mizer, base_lr=self.learning_rate, max_lr=0.001, cycle_momentum=True)
        return [opt_mizer] #, [scheduler]

    def train_dataloader(self):
        # REQUIRED
        return DataLoader(self.train_dataset,
                      self.batch_size,
                      self.shuffle,
                      pin_memory=True, 
                      collate_fn=self.train_dataset.my_collate) 

    def val_dataloader(self):
        # OPTIONAL
        return DataLoader(self.vaild_dataset,
                      self.batch_size,
                      self.shuffle,
                      pin_memory=True, 
                      collate_fn=self.vaild_dataset.my_collate) 
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset,
                      self.batch_size,
                      self.shuffle,
                      pin_memory=True, 
                      collate_fn=self.test_dataset.my_collate) 

def dfs_freeze(model):
    for name, child in model.named_children():
        for param in child.parameters():
            param.requires_grad = False
        dfs_freeze(child)
    model.eval()

def dfs_unfreeze(model):
    model.train()
    for name, child in model.named_children():
        for param in child.parameters():
            param.requires_grad = True
        dfs_freeze(child)
        
def select_by_original_(df):
    cc = df.groupby('original').apply(lambda x: x.sample(n=1))
    cc = [i[1] for i in cc.index]
    return df.loc[cc]


def validation_of_reshape_for_back_cnn():
    
    def test_for_and_back(x, op_instance):     
        B, frame, face, C, H, W = x.size()
        x = x.reshape((B *frame *face, C, H, W))
        # 8 - output channel
        res = op_instance(x).reshape(B, frame*face*8*H*W)
        return res
    
    # init batch size with 2, 3-frames, 5-face, 3-channel, H,W 
    a = torch.randint(2, 100, (2,3, 5,3, 244, 244)).float()
    # a1 is the first sample from the batch 
    a1  = a[0].unsqueeze(0)
    
    # define a single con2d 
    ly = nn.Conv2d(3, 8, 3, padding=1, bias=False)  

    res_a = test_for_and_back(a, ly)
    res_a1 = test_for_and_back(a1, ly)
    print(res_a[0] == res_a1)