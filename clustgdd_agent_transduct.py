import numpy as np
import torch
import torch.nn as nn
import torch.optim as optimac
from torch.nn import Parameter
import torch.nn.functional as F
from utils import match_loss, regularization, row_normalize_tensor
import deep_robust_utils as utils
from deep_robust_utils import sparse_mx_to_torch_sparse_tensor 
from utils_clustgdd import graph_analysis, ER_estimator, attaw_ER_estimator
from copy import deepcopy
import numpy as np
from models.gcn import GCN,MLP
# from models.sgc import SGC 
from models.sgc_multi import SGC as SGC
from sklearn.cluster import KMeans, MiniBatchKMeans
from scipy.sparse import coo_matrix



# decomposed multi-view graph condense
class ClustGDD:
    def __init__(self, data, args, device='cuda', **kwargs):
        self.data = data
        self.args = args
        self.device = device
        self.ori_node_num =  data.feat_full.shape[0]

        n = int(data.feat_train.shape[0] * args.reduction_rate)
        d = data.feat_train.shape[1]
        
        self.nnodes_syn = n
        self.d = d 
        self.args = args

        print('adj_syn:', (n,n), 'feat_syn:', (n,d))
    
    def pretrained_clustering(self,data):
        """subspace clustering on propagated node attributes"""
        # get the raw node feature, adjacency matrix, labels, and train idx 
        features, adj_np , labels = data.feat_full, data.adj_full, data.labels_full
        idx_train = data.idx_train 
        features, adj, labels = utils.to_tensor(features, adj_np, labels, device=self.device)
        ori_edge_num = adj._indices().shape[-1] 
        self.ori_edge_num = ori_edge_num 
        
        # normalized the adjacency matrix
        if utils.is_sparse_tensor(adj):
            adj_norm = utils.normalize_adj_tensor(adj, sparse=True)
        else:
            adj_norm = utils.normalize_adj_tensor(adj) 

        # do feature propagation via the adjacency matrix 
        # T: the propagation number
        T = self.args.prop_num 
        alpha = self.args.alpha 
        
        # the closed form of feature denosing 
        for t in range(T):
            if t == 0:
                prop_feat = features 
                target_feat = (1-alpha)* prop_feat 
            else:
                prop_feat = alpha*adj_norm @ prop_feat
                target_feat = target_feat + (1-alpha)* prop_feat 
            
        # training a MLP/linear to get the better embedding for better cluster
        with_relu = False
        with_bn = False 
        model = MLP(nfeat= prop_feat.shape[-1] ,nhid=self.args.hidden, dropout=self.args.predropout,
                    weight_decay=self.args.prewd, nlayers=self.args.prenlayers, lr=self.args.prelr, with_relu=with_relu, with_bn= with_bn, 
                    nclass=data.nclass, device=self.device).to(self.device) 

        model.fit_with_val(target_feat, adj_norm, labels[idx_train],idx_train, data,
                        train_iters=self.args.preep, normalize=True, verbose=False)
        
        labels_test = torch.LongTensor(data.labels_test).to(self.device)
        labels_train = torch.LongTensor(data.labels_train).to(self.device)
        
        output = model.predict(target_feat)
        loss_train = F.nll_loss(output[idx_train], labels_train)
        acc_train = utils.accuracy(output[idx_train], labels_train)

        print("MLP pretrain, train set results:",
                  "loss= {:.4f}".format(loss_train.item()),
                  "accuracy= {:.4f}".format(acc_train.item()))
        
        with torch.no_grad():
            model.eval()
            _, output = model.predict(target_feat,mode='e')
        loss_test = F.cross_entropy(output[data.idx_test], labels_test)
        acc_test = utils.accuracy(output[data.idx_test], labels_test)
        
        print("MLP pretrain, test set results:",
                "loss= {:.4f}".format(loss_test.item()),
                "accuracy= {:.4f}".format(acc_test.item()))
        
         # do kmeans on the left eigen vector
        
        output_np = output.cpu().numpy() 

        if self.args.dataset == 'ogbn-arxiv':
            cluster = MiniBatchKMeans(n_clusters=self.nnodes_syn, random_state=self.args.seed, batch_size=self.args.cluster_minibatch).fit(output_np)
        else:
            cluster = KMeans(n_clusters = self.nnodes_syn).fit(output_np) 
        cluster_centers = cluster.cluster_centers_
        cluster_labels = cluster.labels_ 

        # self.cluster_measure(cluster_labels)

        print("finish clustering")    
        # get the train nodes representations and labels 
        # get the class-wise train nodes representation centers
        # the cluster center and class center, assign the cluster center a label
        # get the node class from cluster
        cluster_centers = torch.FloatTensor(cluster_centers).to(self.device)
        cluster_labels = torch.FloatTensor(cluster_labels).to(self.device) 
        l0 = [] 
        ft_source = target_feat

        for i in range(self.nnodes_syn): 
            cluster_feat_temp_center = ft_source[torch.where(cluster_labels==i)[0]].mean(dim=0)
            l0.append(cluster_feat_temp_center)

        cluster_feat_centers =  torch.stack(l0,dim=0)
        cluster_center_labels = torch.argmax(cluster_centers,dim=-1) 
        cluster_labels = cluster_labels.int()

        return cluster_feat_centers, cluster_center_labels, cluster_labels, adj_norm, features, adj, labels,target_feat, idx_train, data.idx_val,  output
    
    def graph_sparse(self, adj, ratio,ebd=None , sp_type='vanilla'):
        edge_index = adj.coalesce()._indices() 
        values     = adj.coalesce()._values()
        src = edge_index[0] 
        dst = edge_index[1]
        nedges = values.shape[0]

        if sp_type == 'vanilla':
            ER_low = ER_estimator(adj,src,dst)
            edge_weight = values
            sampled_weight = ER_low 
            sampled_edges_num = int(nedges*ratio)
            topk_values, topk_indices = torch.topk(sampled_weight, sampled_edges_num)
            sampled_edge_index_u = topk_indices

            # construct graph accordding to the index and weight
            sampled_src = src[sampled_edge_index_u].cpu().numpy()
            sampled_dst = dst[sampled_edge_index_u].cpu().numpy()
            values = values[sampled_edge_index_u].cpu().numpy()
            sparsed_graph = coo_matrix((values,(sampled_src,sampled_dst)),shape=adj.shape)
            sparsed_graph = sparse_mx_to_torch_sparse_tensor(sparsed_graph).to(adj.device)
            return [sparsed_graph]

        elif sp_type == 'attaw':
            ER_low, reweighted_graph = attaw_ER_estimator(adj,ebd, src,dst)
            sparsed_graph_list= []
            ebd_stmx = F.softmax(ebd,dim=-1)
            sampled_edges_num = int(nedges*ratio)
            print(nedges)
            print(sampled_edges_num)
            for i in range(ebd_stmx.shape[-1]):
                values  = reweighted_graph.coalesce()._values()
                class_prob = ebd_stmx[:, i]
                src_prob = class_prob[src]
                dst_prob = class_prob[dst]
                sampled_weight = src_prob*dst_prob*ER_low
                topk_values, topk_indices = torch.topk(sampled_weight, sampled_edges_num)
                sampled_edge_index_u = topk_indices
                # construct graph accordding to the index and weight
                sampled_src = src[sampled_edge_index_u].cpu().numpy()
                sampled_dst = dst[sampled_edge_index_u].cpu().numpy()
                
                # current version
                values = values[sampled_edge_index_u].cpu().numpy()

                # tried version
                # values = sampled_weight[sampled_edge_index_u].cpu().numpy()
                sparsed_graph = coo_matrix((values,(sampled_src,sampled_dst)),shape=adj.shape)
                sparsed_graph = sparse_mx_to_torch_sparse_tensor(sparsed_graph).to(adj.device)
                sparsed_graph_list.append(sparsed_graph)

            return sparsed_graph_list
        
        elif sp_type == 'single':
            ER_low, reweighted_graph = attaw_ER_estimator(adj,ebd, src,dst)
            sparsed_graph_list= []
            ebd_stmx = F.softmax(ebd,dim=-1)
            sampled_edges_num = int(nedges*ratio)
            
            values  = reweighted_graph.coalesce()._values()
            sampled_weight = ER_low
            topk_values, topk_indices = torch.topk(sampled_weight, sampled_edges_num)
            sampled_edge_index_u = topk_indices
            # construct graph accordding to the index and weight
            sampled_src = src[sampled_edge_index_u].cpu().numpy()
            sampled_dst = dst[sampled_edge_index_u].cpu().numpy()
            values = values[sampled_edge_index_u].cpu().numpy()
            sparsed_graph = coo_matrix((values,(sampled_src,sampled_dst)),shape=adj.shape)
            sparsed_graph = sparse_mx_to_torch_sparse_tensor(sparsed_graph).to(adj.device)
            sparsed_graph_list.append(sparsed_graph)

            return sparsed_graph_list
              
        elif sp_type == 'no_sp':
            return [adj]
        
        elif sp_type == 'rand':
            ER_low = ER_estimator(adj,src,dst)
            edge_weight = values
            sampled_weight = ER_low 
            sampled_edges_num = int(nedges*ratio)
            topk_values, topk_indices = torch.topk(sampled_weight, sampled_edges_num)
            sparsed_graph_list = []
            for i in range(ebd.shape[-1]):
                values     = adj.coalesce()._values()
                src = edge_index[0] 
                dst = edge_index[1]
                start = 0
                end = nedges
                num_samples = sampled_edges_num
                random_integers = torch.randperm(end - start) + start
                sampled_edge_index_u = random_integers[:num_samples]

                # construct graph accordding to the index and weight
                sampled_src = src[sampled_edge_index_u].cpu().numpy()
                sampled_dst = dst[sampled_edge_index_u].cpu().numpy()
                values = values[sampled_edge_index_u].cpu().numpy()
                sparsed_graph = coo_matrix((values,(sampled_src,sampled_dst)),shape=adj.shape)
                sparsed_graph = sparse_mx_to_torch_sparse_tensor(sparsed_graph).to(adj.device)
                sparsed_graph_list.append(sparsed_graph)

            return sparsed_graph_list
        
    def graph_compress(self, cluster_labels, adj_norm, adj_list):
        cluster_labels = cluster_labels.cpu().numpy()
        cluster_num = cluster_labels.max()+1
        cluster_labels_mat = torch.tensor(np.eye(cluster_num)[cluster_labels]).float().to(adj_list[0].device)
        column_sums = cluster_labels_mat.sum(dim=0, keepdim=True)
        cluster_labels_mat = cluster_labels_mat / column_sums
        compressed_graph_list = []

        for adj in adj_list:
            compressed_graph = (cluster_labels_mat.transpose(0,1) @ adj) @ cluster_labels_mat
            compressed_graph = (compressed_graph - torch.diag(torch.diag(compressed_graph))).to_sparse()
            compressed_graph_list.append(compressed_graph)

        adj_syn = (cluster_labels_mat.transpose(0,1) @ adj_norm) @ cluster_labels_mat
        adj_syn = (adj_syn - torch.diag(torch.diag(adj_syn))).to_sparse()
        
        return compressed_graph_list, adj_syn
    
    def graph_refusion(self,target_feat, idx_train, idx_val, labels_raw, feat_syn, compressed_graph_list, label_syn ):
        adj_syn = 0.
        nclass = labels_raw.max()+1

        T = self.args.prop_num 
        alpha = self.args.alpha 
        frcoe = self.args.frcoe
        csttemp = self.args.csttemp

        feat_syn_refine = nn.Parameter(torch.zeros(feat_syn.shape[0], feat_syn.shape[1]).to(self.device))
        # reweighted_matrix_list = []
        # for i in compressed_graph_list:
        #     reweighted_matrix = nn.Parameter(torch.ones(feat_syn.shape[0], feat_syn.shape[0]).to(self.device))
        #     reweighted_matrix_list.append(reweighted_matrix)
        
        # the closed form of feature denosing
        T = self.args.postprop_num 
        target_feat_syn_list = [] 
        i =0 
        for compressed_graph in compressed_graph_list: 
            for t in range(T):
                if t == 0:
                    prop_feat_syn = (feat_syn+frcoe*feat_syn_refine)
                    target_feat_syn = (1-alpha)* prop_feat_syn 
                else: 
                    prop_feat_syn = alpha* (compressed_graph.to_dense()) @ prop_feat_syn
                    target_feat_syn = target_feat_syn + (1-alpha)* prop_feat_syn  
            target_feat_syn_list.append(target_feat_syn)
            i +=1 

        with_relu = False 
        with_bn = False 
        model = MLP(nfeat= feat_syn.shape[-1] ,nhid=self.args.hidden, dropout= self.args.predropout,
                    weight_decay=self.args.prewd, nlayers=self.args.prenlayers, lr=self.args.prelr, with_relu=with_relu, with_bn= with_bn, 
                    nclass=nclass, device=self.device).to(self.device) 
        
        optimizer_feat = torch.optim.Adam([feat_syn_refine], lr=self.args.postlr_feat,weight_decay=self.args.postwd_feat)
        # optimizer_rwm = torch.optim.Adam(reweighted_matrix_list, lr=self.args.postlr_adj, weight_decay=self.args.postwd_adj)
        optimizer_model = torch.optim.Adam(model.parameters(), lr=self.args.postlr_model,weight_decay=self.args.postwd_model)
        best_acc_val = 0. 

        # if self.args.dataset == 'ogbn-arxiv':
        #     coe1 = (1/(label_syn.max()+1))*0.5 # tuning day:12/13 if not tuning successfully, turn it back
        # else:
        coe1 = self.args.predcoe  
        for i in range(self.args.postep):
            pred_list = []
            optimizer_feat.zero_grad()
            # optimizer_rwm.zero_grad()
            optimizer_model.zero_grad()
            
            pred = model(target_feat)
            pred_list.append(pred)
            for target_feat_syn in target_feat_syn_list:
                pred = model(target_feat_syn)
                pred_list.append(pred)
            
            if i == self.args.postep // 2:
                optimizer_feat = torch.optim.Adam([feat_syn_refine], lr=self.args.postlr_feat * 0.1, weight_decay=self.args.postwd_feat)
                # optimizer_rwm  = torch.optim.Adam(reweighted_matrix_list, lr=self.args.postlr_adj*0.1, weight_decay=self.args.postwd_adj)
                optimizer_model  = torch.optim.Adam(model.parameters(), lr=self.args.postlr_model*0.1,weight_decay=self.args.postwd_model)
            
            loss_train = F.nll_loss(pred_list[0][idx_train], labels_raw[idx_train])

            for j in range(1,len(pred_list)):
                loss_train += coe1*F.nll_loss(pred_list[j],label_syn)

            loss_cst=  self.consistency_loss(pred_list[1:],temp=csttemp)
            loss_all= self.args.w1* loss_cst + self.args.w2*loss_train

            loss_all.backward(retain_graph=True)
            optimizer_model.step()
            optimizer_feat.step()
            # optimizer_rwm.step()

            with torch.no_grad():
                model.eval()
                output = model(target_feat)
                loss_val = F.nll_loss(output[idx_val], labels_raw[idx_val])
                acc_val = utils.accuracy(output[idx_val], labels_raw[idx_val])
                if i % 100 == 0:
                    print('Epoch {}, training loss: {}'.format(i, loss_train.item()))
                    print('Epoch {}, acc val: {}'.format(i, acc_val.item()))
                if acc_val > best_acc_val:
                    best_acc_val = acc_val
                    # best_reweighted_matrix_list = deepcopy(reweighted_matrix_list)
                    best_feat_syn_refine = feat_syn_refine.detach()

        feat_syn = feat_syn + frcoe*best_feat_syn_refine.detach()
        i=0
        # for compressed_graph in compressed_graph_list:
        #     adj_syn +=  compressed_graph.to_dense() *(best_reweighted_matrix_list[i])
        #     i +=1 
        # adj_syn = adj_syn.detach().to_sparse()
        # return feat_syn, adj_syn, label_syn 
        return feat_syn

    def test_with_val(self,runs, verbose=True):
        res = []

        data, device = self.data, self.device
        feat_syn, adj_syn, labels_syn = self.feat_syn.detach(), self.adj_syn, self.labels_syn
        if self.args.notopo:
            adj_syn = torch.eye(feat_syn.shape[0]).to(device)
        with_bn = True if self.args.dataset in ['ogbn-arxiv'] else False
        model = GCN(nfeat=feat_syn.shape[1], nhid=self.args.hidden, dropout=0.5,
                    weight_decay=5e-4, nlayers=2,
                    nclass=data.nclass, device=device).to(device)

        if self.args.dataset in ['ogbn-arxiv']:
            model = GCN(nfeat=feat_syn.shape[1], nhid=self.args.hidden, dropout=0.5,
                        weight_decay=0e-4, nlayers=2, with_bn=with_bn,
                        nclass=data.nclass, device=device).to(device)

        args = self.args

        model.fit_with_val(feat_syn, adj_syn, labels_syn, data,
                     train_iters=600, verbose=False)

        model.eval()
        labels_test = torch.LongTensor(data.labels_test).cuda()

        labels_train = torch.LongTensor(data.labels_train).cuda()
        output = model.predict(data.feat_train, data.adj_train)
        loss_train = F.nll_loss(output, labels_train)
        acc_train = utils.accuracy(output, labels_train)
        if verbose:
            print("Train set results:",
                  "loss= {:.4f}".format(loss_train.item()),
                  "accuracy= {:.4f}".format(acc_train.item()))
        res.append(acc_train.item())

        # Full graph
        output = model.predict(data.feat_full, data.adj_full)
        loss_test = F.nll_loss(output[data.idx_test], labels_test)
        acc_test = utils.accuracy(output[data.idx_test], labels_test)
        res.append(acc_test.item())
        if verbose:
            print("Test set results:",
                  "loss= {:.4f}".format(loss_test.item()),
                  "accuracy= {:.4f}".format(acc_test.item()))
        return res

    def train(self):
        import time
        t1 = time.time() 
        args = self.args 
        data = self.data 
        # do clustering on the pretrained results of graph denoising process
        feat_syn, labels_syn, cluster_labels, adj_norm, features, adj, labels, target_feat, idx_train,idx_val, ebd  = self.pretrained_clustering(data)
        t_pc = time.time()
        # doing the compression by the clustering matrix 
        sparsed_graph_list = self.graph_sparse(adj_norm, ratio=self.args.sp_ratio, ebd=ebd, sp_type=self.args.sp_type)
        compressed_graph_list, adj_syn = self.graph_compress(cluster_labels, adj_norm, sparsed_graph_list)
        feat_syn = self.graph_refusion(target_feat, idx_train, idx_val,  labels, feat_syn, compressed_graph_list, labels_syn )
        t2 = time.time()
        max_memory = torch.cuda.max_memory_allocated(self.device)
        adj_syn =  adj_syn.detach().to_dense() 
        adj_syn_norm = utils.normalize_adj_tensor(adj_syn) 
        
        self.feat_syn = feat_syn 
        self.labels_syn = labels_syn 
        self.adj_syn = adj_syn_norm

        # train a GNN on the distilled graph
        if not self.args.tm_rec:
            res = []
            runs = 5 
            for i in range(runs):
                res.append(self.test_with_val(i))

            res = np.array(res)
            print('Train/Test Mean Accuracy:',
                    repr([res.mean(0), res.std(0)]))
        print(f"The pretraining time is {t_pc-t1}")
        print(f"The refinement time is {t2-t_pc}")
        print("Total time is {}".format(t2-t1))
        print(f"max memory allocation: {max_memory / (1024**2):.2f} MB")
        return 


    def cluster_measure(self,cluster_labels):
        # 计算每个聚类的大小
        cluster_sizes = np.bincount(cluster_labels)

        # 计算标准差和变异系数
        std_dev = np.std(cluster_sizes)
        mean_size = np.mean(cluster_sizes)
        coefficient_of_variation = std_dev / mean_size

        print(f"Cluster Sizes: {cluster_sizes}")
        print(f"Standard Deviation: {std_dev:.4f}")
        print(f"Mean Size: {mean_size:.4f}")
        print(f"Coefficient of Variation: {coefficient_of_variation:.4f}")

    # new version
    def consistency_loss(self,out_list,temp=0.1): 
        ps = [torch.softmax(p,dim=1) for p in out_list]  
        sum_p = 0.
        for p in ps:
            sum_p = sum_p + p
        avg_p = sum_p/len(ps)
        
        sharp_p = (torch.pow(avg_p, 1./temp) / torch.sum(torch.pow(avg_p, 1./temp), dim=1, keepdim=True)).detach()
        loss = 0.
        for p in ps:
            loss += torch.sum( (p-sharp_p).pow(2).sum(1) )
        
        loss = loss/len(ps)
        return temp * loss 
    
    def training_loss(self,out_list, label_syn, nll):
        output = 0. 
        i = 0
        out_list1 = [torch.softmax(p,dim=1) for p in out_list]
        for out in out_list1:
            output = out + output 
            i +=1 
        output = output/i 
        loss = nll(torch.log(output),label_syn)
        return loss


    
