
import numpy as np
import random
import time
import argparse
import torch
from utils import *
from utils_clustgdd import * 
import torch.nn.functional as F
import time 

from clustgdd_agent_induct import ClustGDD
from utils_graphsaint import DataGraphSAINT

parser = argparse.ArgumentParser()
parser.add_argument('--gpu_id', type=int, default=0, help='gpu id')
parser.add_argument('--dataset', type=str, default='cora')

parser.add_argument('--epochs', type=int, default=2000)
parser.add_argument('--nlayers', type=int, default=2)
parser.add_argument('--hidden', type=int, default=256)
parser.add_argument('--weight_decay', type=float, default=0.0)
parser.add_argument('--dropout', type=float, default=0.0)
parser.add_argument('--normalize_features', type=bool, default=True)
parser.add_argument('--keep_ratio', type=float, default=1.0)
parser.add_argument('--reduction_rate', type=float, default=1)
parser.add_argument('--seed', type=int, default=15, help='Random seed.')
parser.add_argument('--sgc', type=int, default=1)
parser.add_argument('--save', type=int, default=0)

#args
parser.add_argument('--gctype', type=str, default='clustgdd')
parser.add_argument('--prop_num',type=int, default=1, help='the steps of feature propagations')
parser.add_argument('--alpha',type=float, default=0.8,help='the prop coe')
parser.add_argument('--prehidden',type=int, default=256,help='the pretraining model hidden dimension')
parser.add_argument('--predropout',type=float, default=0.6,help='the pretraining model hidden dimension')
parser.add_argument('--prewd',type=float, default=5e-4,help='the pretraining model wd')
parser.add_argument('--prelr',type=float, default=0.01,help='the pretraining model lr ')
parser.add_argument('--preep',type=int, default=600,help='pretraining epochs')
parser.add_argument('--prenlayers',type=int, default=2, help='the layers of pretraining linear')
parser.add_argument('--cluster_minibatch',type=int, default=1000, help='if the graph is large, using mini-batch clustering')

parser.add_argument('--sp_ratio',type=float, default=1.0,help='the sampling ratio in the graph sparsification stage')
parser.add_argument('--sp_type',type=str, default='attaw',help='the sparsification type')

parser.add_argument('--postep',type=int, default=100,help='post training epochs')
parser.add_argument('--postprop_num',type=int, default=1,help='post propgation steps')
parser.add_argument('--postlr_feat',type=float, default=1e-4,)
parser.add_argument('--postlr_adj',type=float, default=1e-4,)
parser.add_argument('--postlr_model',type=float, default=1e-2,)
parser.add_argument('--postwd_feat',type=float, default=5e-4,)
parser.add_argument('--postwd_adj',type=float, default=5e-4,)
parser.add_argument('--postwd_model',type=float, default=5e-4,)
parser.add_argument('--predcoe',type=float, default=1.)
parser.add_argument('--frcoe',type=float, default=0.01,)
parser.add_argument('--csttemp',type=float, default=0.5,)
parser.add_argument('--w1', type=float, default=0.1, help='weights of consistency loss')
parser.add_argument('--w2', type=float, default=1., help='weights of training loss on syn_graph')
parser.add_argument('--no_refinement',type=bool, default=False,)

parser.add_argument('--no_adjsyn',type=bool, default=False,)
parser.add_argument('--save_pretrained_output',type=bool, default=False,)
parser.add_argument('--save_syn_output',type=bool, default=False,)
parser.add_argument('--save_norf',type=bool, default=False,)
parser.add_argument('--notopo',type=bool, default=False,)
parser.add_argument('--tm_rec',type=bool, default=False)

args = parser.parse_args()

torch.cuda.set_device(args.gpu_id)

# random seed setting
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

print(args)
device='cuda:{}'.format(args.gpu_id)

data_graphsaint = ['flickr', 'reddit', 'ogbn-arxiv']
if args.dataset in data_graphsaint:
    data = DataGraphSAINT(args.dataset)
    data_full = data.data_full
else:
    data_full = get_dataset(args.dataset, args.normalize_features)
    data = Transd2Ind(data_full, keep_ratio=args.keep_ratio)

if args.gctype == 'clustgdd':
    agent = ClustGDD(data, args, device=device)

agent.train()



