import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
import config
from temporal_ensembling import train
from utils import GaussianNoise, savetime, save_exp
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn import Parameter
import torch.nn.init as nn_init
import numpy as np
#from mean_teacher_archs import *
from pytorch_cifar_resnet import *

class Expression(nn.Module):
    def __init__(self, func):
        super(Expression, self).__init__()
        self.func = func
    
    def forward(self, input):
        return self.func(input)

class CNN(nn.Module):    
    def __init__(self, batch_size, std, p=0.5, fm1=96, fm2=192, fm3=192):
        super(CNN, self).__init__()
        self.fm1   = fm1
        self.fm2   = fm2
        self.fm3   = fm3
        self.std   = std
        self.gn    = GaussianNoise(batch_size, std=self.std)
        self.act   = nn.ReLU()
        self.drop  = nn.Dropout(p)
        self.conv1 = weight_norm(nn.Conv2d(3, self.fm1, 3, padding=1))
        self.conv1a = weight_norm(nn.Conv2d(self.fm1, self.fm1, 3, padding=1))
        self.conv1b = weight_norm(nn.Conv2d(self.fm1, self.fm1, 3, padding=1))
        self.conv2 = weight_norm(nn.Conv2d(self.fm1, self.fm2, 3, padding=1))
        self.conv2a = weight_norm(nn.Conv2d(self.fm2, self.fm2, 3, padding=1))
        self.conv2b = weight_norm(nn.Conv2d(self.fm2, self.fm2, 3, padding=1))
        self.conv3 = weight_norm(nn.Conv2d(self.fm2, self.fm2, 3, padding=1))
        self.conv3a = weight_norm(nn.Conv2d(self.fm2, self.fm2, 3, padding=1))
        self.conv3b = weight_norm(nn.Conv2d(self.fm2, self.fm2, 3, padding=1))
        self.avgnet = nn.Sequential(
            Expression(lambda tensor: tensor.mean(3).mean(2).squeeze()))
        self.mp    = nn.MaxPool2d(3, stride=2, padding=1)
        #self.pool = nn.AvgPool2d(4)
        self.fc    = nn.Linear(self.fm2, 10)
    
    def forward(self, x):
        if self.training:
            x = self.gn(x)
        x = self.act(self.conv1(x))
        x = self.act(self.conv1a(x))
        x = self.act(self.mp(self.conv1b(x)))
        x = self.drop(x)
        x = self.act(self.conv2(x))
        x = self.act(self.conv2a(x))
        x = self.act(self.mp(self.conv2b(x)))
        x = self.drop(x)
        x = self.act(self.conv3(x))
        x = self.act(self.conv3a(x))
        x = self.act(self.mp(self.conv3b(x)))
        #x = x.view(-1, self.fm2 * 4 * 4)
        #x = self.drop(x)
        x = self.avgnet(x)
        #x = self.pool(x)
        x = self.fc(x)
        return x


# metrics
accs         = []
accs_best    = []
losses       = []
sup_losses   = []
unsup_losses = []
idxs         = []


ts = savetime()
cfg = vars(config)

for i in range(cfg['n_exp']):
    #model = CNN(cfg['batch_size'], cfg['std'])
    #model = cifar_shakeshake26()
    model = ResNet18()
    seed = cfg['seeds'][i]
    acc, acc_best, l, sl, usl, indices = train(model, seed, **cfg)
    accs.append(acc)
    accs_best.append(acc_best)
    losses.append(l)
    sup_losses.append(sl)
    unsup_losses.append(usl)
    idxs.append(indices)

print ('saving experiment')    

save_exp(ts, losses, sup_losses, unsup_losses,
         accs, accs_best, idxs, **cfg)

