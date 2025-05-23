import torch.nn as nn
import torch.nn.functional as F
import math
import torch
import torch.optim as optim
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module,BatchNorm1d
from deeprobust.graph import utils
from copy import deepcopy
from sklearn.metrics import f1_score
from torch.nn import init
import torch_sparse
from torch.nn import Linear,Dropout

# here we use
class LinkPredictor(Module):
    def __init__(self, nfeat,ntebd , nhid, nlayers=2, dropout=0.5, lr=0.01, weight_decay=5e-4, device=None, weight= 1. ):
        
        self.device = device
        self.nfeat = nfeat
        self.ntebd = ntebd 
        self.nhid = nhid 
        self.nlayers = nlayers
        self.dropout = dropout 
        self.lr = lr 
        self.weight_decay =weight_decay
       
        self.weight = weight 

        self.linear_x1 = Linear(nfeat, nhid) 
        self.dropout_x1 = Dropout(self.dropout)
        self.bn_x1 = BatchNorm1d(nhid)
        self.act_x1 = nn.ReLU() 
        self.linear_t1 = Linear(ntebd,nhid) 
        self.dropout_t1 = Dropout(self.dropout)
        self.bn_t1 = BatchNorm1d(nhid)
        self.act_t1 = nn.ReLU()

        self.linear_x2 = Linear(nfeat, nhid) 
        self.dropout_x2 = Dropout(self.dropout)
        self.bn_x2 = BatchNorm1d(nhid)
        self.act_x2 = nn.ReLU() 
        self.linear_t2 = Linear(ntebd,nhid) 
        self.dropout_t2 = Dropout(self.dropout)
        self.bn_t2 = BatchNorm1d(nhid)
        self.act_t2 = nn.ReLU()

        self.lino1 =  Linear(2*nhid, 1)
        self.lino2 =  Linear(2*nhid, 1) 

        self.reset_parameters()
    
    def forward(self, x_src, t_src, x_dst, t_dst):

        # given the attribute x and topology embedding t
        predicted_adj = None 

        hx_src = self.act_x1(self.dropout_x1(self.bn_x1(self.linear_x1(x_src))))
        ht_src = self.act_t1(self.dropout_t1(self.bn_t1(self.linear_t1(t_src))))
        h_src = (1.-self.weight)*hx_src + self.weight* ht_src

        hx_dst = self.act_x2(self.dropout_x2(self.bn_x2(self.linear_x2(x_dst))))
        ht_dst = self.act_t1(self.dropout_t2(self.bn_t2(self.linear_t2(t_dst))))
        h_dst = (1.-self.weight)*hx_dst + self.weight* ht_dst 

        ho1 = torch.stack([h_src,h_dst],dim=0)   
        ho2 = torch.stack([h_dst, h_src],dim=0) 
        predicted_adj = (self.lino1(ho1) + self.lino2(ho2))/2 

        return predicted_adj 

    def reset_parameters(self):
        def weight_reset(m):
            if isinstance(m, nn.Linear):
                m.reset_parameters()
            if isinstance(m, nn.BatchNorm1d):
                m.reset_parameters()
        self.apply(weight_reset)