import os.path as osp
import numpy as np
import scipy.sparse as sp
import torch
import torch_geometric.transforms as T
from sklearn.model_selection import train_test_split
from sklearn import metrics
import numpy as np
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from deep_robust_utils import *

from scipy.sparse import  diags
from scipy.linalg import sqrtm 
from scipy.sparse import coo_matrix

import networkx as nx
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.utils import resample



def feat_selection(feat, adj,temp=0.5, lsw = 0.1,  device='cuda:0'):
    edge_index = adj._indices()
    values     = adj._values()
    src = edge_index[0] 
    dst = edge_index[1]
    src = src.cpu().numpy() 
    dst = dst.cpu().numpy()
    values = values.cpu().numpy()

    feat = feat.cpu().numpy()
    nodes_num =  feat.shape[0]
    src_feat = feat[src]
    dst_feat = feat[dst] 

    score = np.exp(-np.linalg.norm((src_feat-dst_feat),axis=-1) /temp )
    score_matrix = coo_matrix((score,(src,dst)),shape=adj.shape)

    ones_vec = np.ones(nodes_num) 
    deg_matrix = diags(score_matrix@ones_vec)
    lap_matrix = deg_matrix -score_matrix

    # transpose the features D X N 
    feat_transpose = feat.transpose()
    value1 = feat_transpose @ deg_matrix @ones_vec
    value2 = (ones_vec.transpose())@ deg_matrix @ ones_vec


    feat1 = (feat_transpose - (value1/(value2+1e-10)).reshape(value1.shape[0],1)).transpose()
    
    value3 = np.diag((feat1.transpose())@lap_matrix@feat1)
    value4 = np.diag((feat1.transpose())@deg_matrix@feat1)       

    ls = value3/(value4+1e-10)
    feat =  feat * (1. + lsw*ls)
    feat = torch.tensor(feat).to(torch.float32).to(device)
    # feat = torch.where(feat > val[:,-1], feat, 0.) 
   
    return feat, ls 

# the motivation of graph decomposition and recondensation 
# need a metric to evaluate the invariant or better things. 

def inter_intra_distance_ratio(feat, labels,dist_metric='cos', device='cuda:0'):
    
    nclass = labels.max().item() + 1 
    nnodes = feat.shape[0]
    labels_onehot = tensor2onehot(labels.cpu()).to(device)
    num_of_each_class = labels_onehot.transpose(0,1) @ (torch.ones(nnodes).to(device))
    normalized_labels_onehot = labels_onehot/(num_of_each_class + 1e-10)
    class_feat_center = normalized_labels_onehot.transpose(0,1) @ feat
    print(class_feat_center)

    # the distance between the raw feat to the class center 
    if dist_metric=='cos':
        feat_norm = torch.linalg.vector_norm(feat, ord=2, dim=-1) 
        feat_center_norm = torch.linalg.vector_norm(class_feat_center, ord=2, dim=-1)
        inter_intra_distance = (feat @ (class_feat_center.transpose(0,1)))/(feat_norm@feat_center_norm +1e-10)
        intra_class = inter_intra_distance*labels_onehot.sum(dim=-1)
        inter_class = (inter_intra_distance.sum(dim=-1)-intra_class)/(nclass-1)
    elif dist_metric == 'eu':
        # inter_intra_distance = torch.linalg.vector_norm( feat - class_feat_center[labels], ord=2,dim=-1)
        # intra_class = inter_intra_distance*labels_onehot.sum(dim=-1)
        # inter_class = (inter_intra_distance.sum(dim=-1)-intra_class)/(nclass-1)
        l = []
        for i in range(nclass):
            temp_class_center =  class_feat_center[i]
            temp_class_centers = temp_class_center.unsqueeze(0).repeat(nnodes, 1)
            sub = torch.linalg.vector_norm( feat - temp_class_centers, ord=2, dim=-1).unsqueeze(1)
            l.append(sub)

        inter_intra_distance = torch.cat(l, dim=1)
        print(inter_intra_distance)
        
        intra_class_distance = (inter_intra_distance * labels_onehot).sum(dim=-1)
        print(intra_class_distance)
  
        inter_class_distance = (inter_intra_distance.sum(dim=-1) - intra_class_distance)/(nclass-1)
    
    ratio = inter_class_distance/(intra_class_distance)
    ratio = ratio.mean()

    return ratio

def calculate_frechet_distance(ebd1,ebd2):
    """torch implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).

    Stable version by Dougal J. Sutherland.

    Params:
    -- mu1   : torch array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.

    Returns:
    --   : The Frechet Distance.
    """
    
    mu1 = torch.mean(ebd1, axis=0)
    sigma1 = torch.tensor(np.cov(ebd1.cpu().detach().numpy(), rowvar=False)).to(mu1.device)
    sigma1 += torch.eye(sigma1.shape[0]).to(mu1.device) * 1e-6
    mu2 = torch.mean(ebd2, axis=0)
    sigma2 = torch.tensor(np.cov(ebd2.cpu().detach().numpy(), rowvar=False)).to(mu1.device)
    sigma2 += torch.eye(sigma2.shape[0]).to(mu1.device) * 1e-6
    diff = mu1 - mu2
    covmean, _ = sqrtm( (sigma1 @ sigma2).cpu().numpy() , disp=False)
    # print(covmean)
    covmean = torch.tensor(covmean).to(mu1.device)
    if covmean.type() == 'torch.cuda.ComplexDoubleTensor':
        covmean = covmean.real
        # covmean = torch.abs(covmean)
    tr_covmean = torch.trace(covmean)
    fid = diff.dot(diff) + torch.trace(sigma1) + torch.trace(sigma2) - 2 * tr_covmean
    mean_diff = diff.dot(diff)
    # print(torch.trace(sigma1),torch.trace(sigma2),tr_covmean)
    cov_trace = + torch.trace(sigma1) + torch.trace(sigma2) - 2 * tr_covmean
    return  fid, mean_diff, cov_trace

# the effective resistance sparsifier
def ER_estimator(adj,src,dst):
    nnodes = adj.shape[0] 
    one_vec = torch.ones(nnodes).to(adj.device)
    degree = adj @ one_vec
    values = adj.coalesce().values()
    src_deg = degree[src] 
    dst_deg = degree[dst] 
    # the lower bound of effective resistance
    ER_lower = (values/(src_deg))+(values/(dst_deg))

    return ER_lower

# class wise 
def attaw_ER_estimator(adj, ebd, src, dst):
    nnodes = adj.shape[0] 
    values = adj.coalesce()._values()
    ebd_src = ebd[src].cpu()
    ebd_dst = ebd[dst].cpu()
    src_np = src.cpu().numpy()
    dst_np = dst.cpu().numpy()
    ebd_smilarity = F.cosine_similarity(ebd_src, ebd_dst, dim=-1).to(adj.device)
    values = values*ebd_smilarity
    reweighted_graph = coo_matrix((values.cpu().numpy(),(src_np,dst_np)),shape=adj.shape)
    reweighted_graph = sparse_mx_to_torch_sparse_tensor(reweighted_graph).to(adj.device)

    one_vec = torch.ones(nnodes).to(adj.device)
    degree = reweighted_graph @ one_vec
    values = reweighted_graph.coalesce().values()
    src_deg = degree[src] 
    dst_deg = degree[dst] 
    # the lower bound of effective resistance
    ER_lower = (values/(src_deg))+(values/(dst_deg)), reweighted_graph

    return ER_lower


    # edge = dataset.data.edge_index
    # edges_num = edge.shape[1]
    # nnodes = dataset.data.x.size()[0]
    # src, dst = edge[0], edge[1]
    # node_degree_list = []    
    
    # for i in range(nnodes):
    #     src_ngh = set( dst[torch.where(src==i)[0]].numpy().tolist())
    #     dst_ngh = set( src[torch.where(dst==i)[0]].numpy().tolist())
    #     ngh = src_ngh | dst_ngh 
    #     node_degree_list.append(len(ngh))
    
    # node_degree = torch.tensor(node_degree_list)
    # ngh_degree_list  = []
    
    # # lower bound of effective resistance
    # ER_low =  (1./node_degree[src] + 1./node_degree[dst])/2    
    
    # return ER_low 


def print_memory_usage(stage):
    print(f"{stage} - current memory usage: {torch.cuda.memory_allocated() / (1024**2)} MB")
    print(f"{stage} - max memory usage: {torch.cuda.max_memory_allocated() / (1024**2)} MB")
    # torch.cuda.reset_max_memory_allocated()

def graph_analysis(adj, feat, label, class_nums=1):

    # calculate the num of edges and nodes
    node_num = feat.shape[0]
    edge_num = adj.coalesce()._indices().shape[-1]
    edge_vs_node = edge_num/node_num

    # calculate distribution of class ratio 
    class_ratio_list = []
    for i in range(class_nums):
        class_ratio_list.append(round(torch.where(label==i)[0].shape[0]/node_num,4))

    # calculate the node homophily
    edge_index = adj.coalesce()._indices() 
    values     = adj.coalesce()._values()

    ones_values = torch.ones_like(values)
    adj_ones = torch.sparse_coo_tensor(edge_index, ones_values, adj.shape)
    label_onehot = F.one_hot(label,num_classes= label.max()+1).to(torch.float32)
    node_homo =   (adj_ones @ label_onehot)
    if edge_num > 100000:
        n = node_homo.size(0)
        indices = torch.arange(n).unsqueeze(0).expand(2, n).to(node_homo.device)
        sparse_diag_matrix = torch.sparse.FloatTensor(indices, 1./(node_homo.sum(-1)+1e-10), torch.Size([n, n])).to(node_homo.device)
        node_homo = sparse_diag_matrix @ node_homo 
    else:
        node_homo = ( torch.diag(1./(node_homo.sum(-1)+1e-10))@ node_homo)
    node_homo = node_homo[torch.arange(node_homo.size(0)), label].mean().item()

    # calculate the edge homophily and the confusion matrix 
    src = edge_index[0] 
    dst = edge_index[1]
    src_label = label[src]
    dst_label = label[dst]
    mask = torch.where(src_label==dst_label ,1.0,0.0).to(values.device)
    mask2 = torch.where(src==dst ,0.0,1.0).to(values.device)
    edge_homo = (values*mask*mask2).sum()/ ((values*mask2).sum())

    return node_num, edge_num, edge_vs_node, node_homo, edge_homo, class_ratio_list
