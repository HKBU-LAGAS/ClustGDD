import torch.nn as nn
import torch.nn.functional as F
import math
import torch
import torch.optim as optim
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import deep_robust_utils as utils
from copy import deepcopy
from sklearn.metrics import f1_score
from torch.nn import init
import torch_sparse
import numpy as np
import scipy.sparse as sp
from torch_geometric.nn import ChebConv


class GraphConvolution(Module):
    """Simple GCN layer, similar to https://github.com/tkipf/pygcn
    """

    def __init__(self, in_features, out_features, with_bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        self.bias = Parameter(torch.FloatTensor(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.T.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        """ Graph Convolutional Layer forward function
        """
        if input.data.is_sparse:
            support = torch.spmm(input, self.weight)
        else:
            support = torch.mm(input, self.weight)
        if isinstance(adj, torch_sparse.SparseTensor):
            output = torch_sparse.matmul(adj, support)
        else:
            output = torch.spmm(adj, support)

        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class GCN(nn.Module):

    def __init__(self, nfeat, nhid, nclass, nlayers=2, dropout=0.5, lr=0.01, weight_decay=5e-4,
            with_relu=True, with_bias=True, with_bn=False, device=None):

        super(GCN, self).__init__()

        assert device is not None, "Please specify 'device'!"
        self.device = device
        self.nfeat = nfeat
        self.nclass = nclass

        self.layers = nn.ModuleList([])

        if nlayers == 1:
            self.layers.append(GraphConvolution(nfeat, nclass, with_bias=with_bias))
        else:
            if with_bn:
                self.bns = torch.nn.ModuleList()
                self.bns.append(nn.BatchNorm1d(nhid))
            self.layers.append(GraphConvolution(nfeat, nhid, with_bias=with_bias))
            for i in range(nlayers-2):
                self.layers.append(GraphConvolution(nhid, nhid, with_bias=with_bias))
                if with_bn:
                    self.bns.append(nn.BatchNorm1d(nhid))
            self.layers.append(GraphConvolution(nhid, nclass, with_bias=with_bias))

        self.dropout = dropout
        self.lr = lr
        if not with_relu:
            self.weight_decay = 0
        else:
            self.weight_decay = weight_decay
        self.with_relu = with_relu
        self.with_bn = with_bn
        self.with_bias = with_bias
        self.output = None
        self.best_model = None
        self.best_output = None
        self.adj_norm = None
        self.features = None
        self.multi_label = None

    def forward(self, x, adj):
        for ix, layer in enumerate(self.layers):
            x = layer(x, adj)
            if ix != len(self.layers) - 1:
                x = self.bns[ix](x) if self.with_bn else x
                if self.with_relu:
                    x = F.relu(x)
                x = F.dropout(x, self.dropout, training=self.training)

        if self.multi_label:
            return torch.sigmoid(x)
        else:
            return F.log_softmax(x, dim=1)
    
    def forward_ebd(self, x, adj,norm='minmax'):
        for ix, layer in enumerate(self.layers):
            x = layer(x, adj)
            if ix != len(self.layers) - 1:
                x = self.bns[ix](x) if self.with_bn else x
                if self.with_relu:
                    x = F.relu(x)
                x = F.dropout(x, self.dropout, training=self.training)
        min_val = x.min(dim=1, keepdim=True)[0]
        max_val = x.max(dim=1, keepdim=True)[0]
        # normalize x
        if norm == 'minmax':
            x = (x-min_val)/(max_val-min_val)
            return x 
        elif norm == 'softmax':
            return F.softmax(x, dim=1) 
        elif norm == 'L2':
            x = F.normalize(x, p=2, dim=1)
            return x 
        elif norm == 'sum':
            x1 = torch.sum(torch.abs(x), dim=1, keepdim=True)
            x = x / x1
            return x
        elif norm == 'zero':
            # 计算每一行的均值和标准差
            row_mean = torch.mean(x, dim=1, keepdim=True)
            row_std = torch.std(x, dim=1, keepdim=True)
            x = (x - row_mean) / row_std
            return x 
        else:
            return x

    def forward_sampler(self, x, adjs):
        # for ix, layer in enumerate(self.layers):
        for ix, (adj, _, size) in enumerate(adjs):
            x = self.layers[ix](x, adj)
            if ix != len(self.layers) - 1:
                x = self.bns[ix](x) if self.with_bn else x
                if self.with_relu:
                    x = F.relu(x)
                x = F.dropout(x, self.dropout, training=self.training)

        if self.multi_label:
            return torch.sigmoid(x)
        else:
            return F.log_softmax(x, dim=1)

    def forward_sampler_syn(self, x, adjs):
        for ix, (adj) in enumerate(adjs):
            x = self.layers[ix](x, adj)
            if ix != len(self.layers) - 1:
                x = self.bns[ix](x) if self.with_bn else x
                if self.with_relu:
                    x = F.relu(x)
                x = F.dropout(x, self.dropout, training=self.training)

        if self.multi_label:
            return torch.sigmoid(x)
        else:
            return F.log_softmax(x, dim=1)

    def initialize(self):
        """Initialize parameters of GCN.
        """
        for layer in self.layers:
            layer.reset_parameters()
        if self.with_bn:
            for bn in self.bns:
                bn.reset_parameters()

    def fit(self, features, adj, labels, idx_train, idx_val=None, train_iters=200, initialize=True, verbose=False, normalize=True, patience=None, **kwargs):

        if initialize:
            self.initialize()

        # features, adj, labels = data.feat_train, data.adj_train, data.labels_train
        if type(adj) is not torch.Tensor:
            features, adj, labels = utils.to_tensor(features, adj, labels, device=self.device)
        else:
            features = features.to(self.device)
            adj = adj.to(self.device)
            labels = labels.to(self.device)

        if normalize:
            if utils.is_sparse_tensor(adj):
                adj_norm = utils.normalize_adj_tensor(adj, sparse=True)
            else:
                adj_norm = utils.normalize_adj_tensor(adj)
        else:
            adj_norm = adj

        if 'feat_norm' in kwargs and kwargs['feat_norm']:
            from utils import row_normalize_tensor
            features = row_normalize_tensor(features-features.min())

        self.adj_norm = adj_norm
        self.features = features

        if len(labels.shape) > 1:
            self.multi_label = True
            self.loss = torch.nn.BCELoss()
        else:
            self.multi_label = False
            self.loss = F.nll_loss

        labels = labels.float() if self.multi_label else labels
        self.labels = labels


        if idx_val is not None:
            self._train_with_val2(labels, idx_train, idx_val, train_iters, verbose)
        else:
            self._train_without_val2(labels, idx_train, train_iters, verbose)

    def _train_without_val2(self, labels, idx_train, train_iters, verbose):
        self.train()
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        for i in range(train_iters):
            if i == train_iters // 2:
                lr = self.lr*0.1
                optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=self.weight_decay)

            optimizer.zero_grad()
            output = self.forward(self.features, self.adj_norm)
            loss_train = self.loss(output[idx_train], labels[idx_train])
            loss_train.backward()
            optimizer.step()
            if verbose and i % 10 == 0:
                print('Epoch {}, training loss: {}'.format(i, loss_train.item()))

        self.eval()
        output = self.forward(self.features, self.adj_norm)
        self.output = output

    def fit_with_val(self, features, adj, labels, data, train_iters=200, initialize=True, 
                     verbose=False, normalize=True, patience=None, noval=False,full=False, **kwargs):
        '''data: full data class'''
        if initialize:
            self.initialize()

        if type(adj) is not torch.Tensor:
            features, adj, labels = utils.to_tensor(features, adj, labels, device=self.device)
        else:
            features = features.to(self.device)
            adj = adj.to(self.device)
            labels = labels.to(self.device)

        if normalize:
            if utils.is_sparse_tensor(adj):
                adj_norm = utils.normalize_adj_tensor(adj, sparse=True)
            else:
                adj_norm = utils.normalize_adj_tensor(adj)   
        else:
            adj_norm = adj

        if 'feat_norm' in kwargs and kwargs['feat_norm']:
            from utils import row_normalize_tensor
            features = row_normalize_tensor(features-features.min())

        self.adj_norm = adj_norm

        self.features = features

        if len(labels.shape) > 1:
            self.multi_label = True
            self.loss = torch.nn.BCELoss()
        else:
            self.multi_label = False
            self.loss = F.nll_loss

        labels = labels.float() if self.multi_label else labels
        self.labels = labels

        if noval:
            self._train_with_val(labels, data, train_iters, verbose, adj_val=True,full=full)
        else:
            self._train_with_val(labels, data, train_iters, verbose,full= full)

    def _train_with_val(self, labels, data, train_iters, verbose, adj_val=False, full= False ):
        if adj_val:
            feat_full, adj_full = data.feat_val, data.adj_val
        else:
            feat_full, adj_full = data.feat_full, data.adj_full
        feat_full, adj_full = utils.to_tensor(feat_full, adj_full, device=self.device)
        adj_full_norm = utils.normalize_adj_tensor(adj_full, sparse=True)
        labels_val = torch.LongTensor(data.labels_val).to(self.device)

        if verbose:
            print('=== training gcn model ===')
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        best_acc_val = 0
        weights = deepcopy(self.state_dict())
        for i in range(train_iters):
            if i == train_iters // 2:
                lr = self.lr*0.1
                optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=self.weight_decay)

            self.train()
            optimizer.zero_grad()
            output = self.forward(self.features, self.adj_norm)
            if full:
                loss_train = self.loss(output[data.idx_train], labels)
            else:
                loss_train = self.loss(output, labels)

            loss_train.backward()
            optimizer.step()

            with torch.no_grad():
                self.eval()
                output = self.forward(feat_full, adj_full_norm)

                if adj_val:
                    loss_val = F.nll_loss(output, labels_val)
                    acc_val = utils.accuracy(output, labels_val)
                else:
                    loss_val = F.nll_loss(output[data.idx_val], labels_val)
                    acc_val = utils.accuracy(output[data.idx_val], labels_val)
                
                if acc_val > best_acc_val:
                    best_acc_val = acc_val
                    self.output = output
                    weights = deepcopy(self.state_dict())
            
        if verbose:
            print('=== picking the best model according to the performance on validation ===')
        self.load_state_dict(weights)

    def test(self, idx_test):
        """Evaluate GCN performance on test set.
        Parameters
        ----------
        idx_test :
            node testing indices
        """
        self.eval()
        output = self.predict()
        # output = self.output
        loss_test = F.nll_loss(output[idx_test], self.labels[idx_test])
        acc_test = utils.accuracy(output[idx_test], self.labels[idx_test])
        print("Test set results:",
              "loss= {:.4f}".format(loss_test.item()),
              "accuracy= {:.4f}".format(acc_test.item()))
        return acc_test.item()

    @torch.no_grad()
    def predict(self, features=None, adj=None):
        """By default, the inputs should be unnormalized adjacency
        Parameters
        ----------
        features :
            node features. If `features` and `adj` are not given, this function will use previous stored `features` and `adj` from training to make predictions.
        adj :
            adjcency matrix. If `features` and `adj` are not given, this function will use previous stored `features` and `adj` from training to make predictions.
        Returns
        -------
        torch.FloatTensor
            output (log probabilities) of GCN
        """

        self.eval()
        if features is None and adj is None:
            return self.forward(self.features, self.adj_norm)
        else:
            if type(adj) is not torch.Tensor:
                features, adj = utils.to_tensor(features, adj, device=self.device)

            self.features = features
            if utils.is_sparse_tensor(adj):
                self.adj_norm = utils.normalize_adj_tensor(adj, sparse=True)
            else:
                self.adj_norm = utils.normalize_adj_tensor(adj)
            return self.forward(self.features, self.adj_norm)
    
    @torch.no_grad()
    def predict_unnorm(self, features=None, adj=None):
        self.eval()
        if features is None and adj is None:
            return self.forward(self.features, self.adj_norm)
        else:
            if type(adj) is not torch.Tensor:
                features, adj = utils.to_tensor(features, adj, device=self.device)

            self.features = features
            self.adj_norm = adj
            return self.forward(self.features, self.adj_norm)

    def _train_with_val2(self, labels, idx_train, idx_val, train_iters, verbose):
        if verbose:
            print('=== training gcn model ===')
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        best_loss_val = 100
        best_acc_val = 0

        for i in range(train_iters):
            if i == train_iters // 2:
                lr = self.lr*0.1
                optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=self.weight_decay)

            self.train()
            optimizer.zero_grad()
            output = self.forward(self.features, self.adj_norm)
            loss_train = F.nll_loss(output[idx_train], labels[idx_train])
            loss_train.backward()
            optimizer.step()

            if verbose and i % 10 == 0:
                print('Epoch {}, training loss: {}'.format(i, loss_train.item()))

            self.eval()
            output = self.forward(self.features, self.adj_norm)
            loss_val = F.nll_loss(output[idx_val], labels[idx_val])
            acc_val = utils.accuracy(output[idx_val], labels[idx_val])

            if acc_val > best_acc_val:
                best_acc_val = acc_val
                self.output = output
                weights = deepcopy(self.state_dict())

        if verbose:
            print('=== picking the best model according to the performance on validation ===')
        self.load_state_dict(weights)

class MLP(nn.Module):

    def __init__(self, nfeat, nhid, nclass, nlayers=2, dropout=0.5, lr=0.01, weight_decay=5e-4,
            with_relu=True, with_bias=True, with_bn=False, device=None):

        super(MLP, self).__init__()

        assert device is not None, "Please specify 'device'!"
        self.device = device
        self.nfeat = nfeat
        self.nclass = nclass

        self.layers = nn.ModuleList([])

        if nlayers == 1:
            self.layers.append(nn.Linear(nfeat, nclass))
        else:
            if with_bn:
                self.bns = torch.nn.ModuleList()
                self.bns.append(nn.BatchNorm1d(nhid))
            self.layers.append(nn.Linear(nfeat, nhid))
            for i in range(nlayers-2):
                self.layers.append(nn.Linear(nhid, nhid))
                if with_bn:
                    self.bns.append(nn.BatchNorm1d(nhid))
            self.layers.append(nn.Linear(nhid, nclass))

        self.dropout = dropout
        self.lr = lr
        if not with_relu:
            self.weight_decay = 0
        else:
            self.weight_decay = weight_decay
        self.with_relu = with_relu
        self.with_bn = with_bn
        self.with_bias = with_bias
        self.output = None
        self.best_model = None
        self.best_output = None
        self.adj_norm = None
        self.features = None
        self.multi_label = None

    def forward(self, x, mode='t'):
        for ix, layer in enumerate(self.layers):
            x = layer(x)
            if ix != len(self.layers) - 1:
                x = self.bns[ix](x) if self.with_bn else x
                if self.with_relu:
                    x = F.relu(x)
                x = F.dropout(x, self.dropout, training=self.training)

        if mode == 't':
            if self.multi_label:
                return torch.sigmoid(x)
            else:
                return F.log_softmax(x, dim=1)
        else:
            if self.multi_label:
                return torch.sigmoid(x),x
            else:
                return F.log_softmax(x, dim=1),x


    def initialize(self):
        """Initialize parameters of GCN.
        """
        for layer in self.layers:
            layer.reset_parameters()
        if self.with_bn:
            for bn in self.bns:
                bn.reset_parameters()

#    target_feat, adj_norm, labels[idx_train],idx_train, data,
#                      train_iters=600, normalize=True, verbose=False
    def fit_with_val(self, features, adj, labels, idx_train, data, train_iters=200, initialize=True, verbose=False, normalize=True, patience=None, noval=False, **kwargs):
        '''data: full data class'''
        if initialize:
            self.initialize()

        if type(adj) is not torch.Tensor:
            features, adj, labels = utils.to_tensor(features, adj, labels, device=self.device)
        else:
            features = features.to(self.device)
            adj = adj.to(self.device)
            labels = labels.to(self.device)

        if normalize:
            if utils.is_sparse_tensor(adj):
                adj_norm = utils.normalize_adj_tensor(adj, sparse=True)
            else:
                adj_norm = utils.normalize_adj_tensor(adj)
        else:
            adj_norm = adj

        if 'feat_norm' in kwargs and kwargs['feat_norm']:
            from utils import row_normalize_tensor
            features = row_normalize_tensor(features-features.min())

        self.adj_norm = adj_norm
        self.features = features

        if len(labels.shape) > 1:
            self.multi_label = True
            self.loss = torch.nn.BCELoss()
        else:
            self.multi_label = False
            self.loss = F.nll_loss

        labels = labels.float() if self.multi_label else labels
        self.labels = labels

        if noval:
            self._train_with_val(labels,idx_train, data, train_iters, verbose, adj_val=True)
        else:
            self._train_with_val(labels,idx_train, data, train_iters, verbose)

    def _train_with_val(self, labels,idx_train, data, train_iters, verbose, adj_val=False):
        if adj_val:
            feat_full, adj_full = data.feat_val, data.adj_val
        else:
            feat_full, adj_full = data.feat_full, data.adj_full
        feat_full, adj_full = utils.to_tensor(feat_full, adj_full, device=self.device)
        adj_full_norm = utils.normalize_adj_tensor(adj_full, sparse=True)
        labels_val = torch.LongTensor(data.labels_val).to(self.device)

        if verbose:
            print('=== training gcn model ===')
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        best_acc_val = 0
        weights = deepcopy(self.state_dict())
        for i in range(train_iters):
            if i == train_iters // 2:
                lr = self.lr*0.1
                optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=self.weight_decay)

            self.train()
            optimizer.zero_grad()
            output = self.forward(self.features)
            loss_train = self.loss(output[idx_train], labels)
            loss_train.backward()
            optimizer.step()

            if i % 100 == 0:
                print('Epoch {}, training loss: {}'.format(i, loss_train.item()))

            with torch.no_grad():
                self.eval()
                output = self.forward(self.features)

                if adj_val:
                    loss_val = F.nll_loss(output, labels_val)
                    acc_val = utils.accuracy(output, labels_val)
                else:
                    loss_val = F.nll_loss(output[data.idx_val], labels_val)
                    acc_val = utils.accuracy(output[data.idx_val], labels_val)

                if acc_val > best_acc_val:
                    best_acc_val = acc_val
                    self.output = output
                    weights = deepcopy(self.state_dict())

        if verbose:
            print('=== picking the best model according to the performance on validation ===')
        self.load_state_dict(weights)


    def test(self, idx_test):
        """Evaluate GCN performance on test set.
        Parameters
        ----------
        idx_test :
            node testing indices
        """
        self.eval()
        output = self.predict()
        # output = self.output
        loss_test = F.nll_loss(output[idx_test], self.labels[idx_test])
        acc_test = utils.accuracy(output[idx_test], self.labels[idx_test])
        print("Test set results:",
              "loss= {:.4f}".format(loss_test.item()),
              "accuracy= {:.4f}".format(acc_test.item()))
        return acc_test.item()


    @torch.no_grad()
    def predict(self, features=None, adj=None,mode='t'):
        """By default, the inputs should be unnormalized adjacency
        Parameters
        ----------
        features :
            node features. If `features` and `adj` are not given, this function will use previous stored `features` and `adj` from training to make predictions.
        adj :
            adjcency matrix. If `features` and `adj` are not given, this function will use previous stored `features` and `adj` from training to make predictions.
        Returns
        -------
        torch.FloatTensor
            output (log probabilities) of GCN
        """

        self.eval()
        if features is None and adj is None:
            return self.forward(self.features,mode)
        else:
            # if type(adj) is not torch.Tensor:
            #     features, adj = utils.to_tensor(features, adj, device=self.device)

            self.features = features
            # if utils.is_sparse_tensor(adj):
            #     self.adj_norm = utils.normalize_adj_tensor(adj, sparse=True)
            # else:
            #     self.adj_norm = utils.normalize_adj_tensor(adj)
            return self.forward(self.features,mode)

    @torch.no_grad()
    def predict_unnorm(self, features=None, adj=None):
        self.eval()
        if features is None and adj is None:
            return self.forward(self.features)
        else:
            if type(adj) is not torch.Tensor:
                features, adj = utils.to_tensor(features, adj, device=self.device)

            self.features = features
            self.adj_norm = adj
            return self.forward(self.features)


    def _train_with_val2(self, labels, idx_train, idx_val, train_iters, verbose):
        if verbose:
            print('=== training gcn model ===')
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        best_loss_val = 100
        best_acc_val = 0

        for i in range(train_iters):
            if i == train_iters // 2:
                lr = self.lr*0.1
                optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=self.weight_decay)

            self.train()
            optimizer.zero_grad()
            output = self.forward(self.features)
            loss_train = F.nll_loss(output[idx_train], labels[idx_train])
            loss_train.backward()
            optimizer.step()

            if verbose and i % 10 == 0:
                print('Epoch {}, training loss: {}'.format(i, loss_train.item()))

            self.eval()
            output = self.forward(self.features)
            loss_val = F.nll_loss(output[idx_val], labels[idx_val])
            acc_val = utils.accuracy(output[idx_val], labels[idx_val])

            if acc_val > best_acc_val:
                best_acc_val = acc_val
                weights = deepcopy(self.state_dict())

        if verbose:
            print('=== picking the best model according to the performance on validation ===')
        self.load_state_dict(weights)

class MLP_Induct(nn.Module):

    def __init__(self, nfeat, nhid, nclass, nlayers=2, dropout=0.5, lr=0.01, weight_decay=5e-4,
            with_relu=True, with_bias=True, with_bn=False, device=None):

        super(MLP_Induct, self).__init__()

        assert device is not None, "Please specify 'device'!"
        self.device = device
        self.nfeat = nfeat
        self.nclass = nclass

        self.layers = nn.ModuleList([])

        if nlayers == 1:
            self.layers.append(nn.Linear(nfeat, nclass))
        else:
            if with_bn:
                self.bns = torch.nn.ModuleList()
                self.bns.append(nn.BatchNorm1d(nhid))
            self.layers.append(nn.Linear(nfeat, nhid))
            for i in range(nlayers-2):
                self.layers.append(nn.Linear(nhid, nhid))
                if with_bn:
                    self.bns.append(nn.BatchNorm1d(nhid))
            self.layers.append(nn.Linear(nhid, nclass))

        self.dropout = dropout
        self.lr = lr
        if not with_relu:
            self.weight_decay = 0
        else:
            self.weight_decay = weight_decay
        self.with_relu = with_relu
        self.with_bn = with_bn
        self.with_bias = with_bias
        self.multi_label = False 


    def forward(self, x, mode='t'):
        # x = F.dropout(x, 0.1, training=self.training)
        for ix, layer in enumerate(self.layers):
            x = layer(x)
            if ix != len(self.layers) - 1:
                x = self.bns[ix](x) if self.with_bn else x
                if self.with_relu:
                    x = F.relu(x)
                x = F.dropout(x, self.dropout, training=self.training)

        if mode == 't':
            if self.multi_label:
                return torch.sigmoid(x)
            else:
                return F.log_softmax(x, dim=1)
        else:
            if self.multi_label:
                return torch.sigmoid(x), x
            else:
                return F.log_softmax(x, dim=1),x


    def initialize(self):
        """Initialize parameters of GCN.
        """
        for layer in self.layers:
            layer.reset_parameters()
        if self.with_bn:
            for bn in self.bns:
                bn.reset_parameters()

    def fit_with_val(self, feat_train, labels_train, 
                     feat_val,labels_val, 
                     train_iters=200, initialize=True, **kwargs):
        '''data: full data class'''
        if initialize:
            self.initialize()

        feat_train = feat_train.to(self.device)
        labels_train = labels_train.to(self.device)

        feat_val = feat_val.to(self.device)
        labels_val = labels_val.to(self.device)

        if len(labels_train.shape) > 1:
            self.multi_label = True
            self.loss = torch.nn.BCELoss()
        else:
            self.multi_label = False
            self.loss = F.nll_loss

        labels_train = labels_train.float() if self.multi_label else labels_train
        labels_val = labels_val.float() if self.multi_label else labels_val

        self._train_with_val(feat_train,labels_train, 
                        feat_val,labels_val, 
                        train_iters,)

    def _train_with_val(self, feat_train,labels_train, 
                        feat_val,labels_val, 
                        train_iters, verbose=False):

        if verbose:
            print('=== training gcn model ===')
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        best_acc_val = 0
        weights = deepcopy(self.state_dict())
        for i in range(train_iters):
            if i == train_iters // 2:
                lr = self.lr*0.1
                optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=self.weight_decay)

            self.train()
            optimizer.zero_grad()
            output = self.forward(feat_train)
            loss_train = self.loss(output, labels_train)
            loss_train.backward()
            optimizer.step()

            # if i % 100 == 0:
            #     print('Epoch {}, training loss: {}'.format(i, loss_train.item()))

            with torch.no_grad():
                self.eval()
                output = self.forward(feat_val)
                acc_val = utils.accuracy(output, labels_val)
                if acc_val > best_acc_val:
                    best_acc_val = acc_val
                    self.output = output
                    weights = deepcopy(self.state_dict())

        if verbose:
            print('=== picking the best model according to the performance on validation ===')
        self.load_state_dict(weights)

    def test(self, feat_test, labels_test):
        """Evaluate GCN performance on test set.
        Parameters
        ----------
        idx_test :
            node testing indices
        """
        self.eval()
        output = self.forward(feat_test)
        # output = self.output
        loss_test = F.nll_loss(output, labels_test)
        acc_test = utils.accuracy(output, labels_test)
        print("Test set results:",
              "loss= {:.4f}".format(loss_test.item()),
              "accuracy= {:.4f}".format(acc_test.item()))
        return acc_test.item()


    @torch.no_grad()
    def predict(self, feat_test, mode='t'):
        self.eval()
        return self.forward(feat_test,mode)
    
class GCNAb(nn.Module):

    def __init__(self, nfeat, nhid, nclass, 
                 nlayers=2, dropout=0.5, lr=0.01, weight_decay=5e-4,
            with_relu=True, with_bias=True, with_bn=False, device=None):

        super(GCNAb, self).__init__()

        assert device is not None, "Please specify 'device'!"
        self.device = device
        self.nfeat = nfeat
        self.nclass = nclass

        self.layers = nn.ModuleList([])

        if nlayers == 1:
            self.layers.append(GraphConvolution(nfeat, nclass, with_bias=with_bias))
        else:
            if with_bn:
                self.bns = torch.nn.ModuleList()
                self.bns.append(nn.BatchNorm1d(nhid))
            self.layers.append(GraphConvolution(nfeat, nhid, with_bias=with_bias))
            for i in range(nlayers-2):
                self.layers.append(GraphConvolution(nhid, nhid, with_bias=with_bias))
                if with_bn:
                    self.bns.append(nn.BatchNorm1d(nhid))
            self.layers.append(GraphConvolution(nhid, nclass, with_bias=with_bias))

        self.dropout = dropout
        self.lr = lr
        if not with_relu:
            self.weight_decay = 0
        else:
            self.weight_decay = weight_decay
        self.with_relu = with_relu
        self.with_bn = with_bn
        self.with_bias = with_bias
        self.output = None
        self.best_model = None
        self.best_output = None
        self.adj_norm = None
        self.features = None
        self.multi_label = None

    def forward(self, x, adj,mode='t'):
        for ix, layer in enumerate(self.layers):
            x = layer(x, adj)
            if ix != len(self.layers) - 1:
                x = self.bns[ix](x) if self.with_bn else x
                if self.with_relu:
                    x = F.relu(x)
                x = F.dropout(x, self.dropout, training=self.training)

        if mode == 't':
            if self.multi_label:
                return torch.sigmoid(x)
            else:
                return F.log_softmax(x, dim=1)
        else:
            if self.multi_label:
                return torch.sigmoid(x), x
            else:
                return F.log_softmax(x, dim=1),x

    def initialize(self):
        """Initialize parameters of GCN.
        """
        for layer in self.layers:
            layer.reset_parameters()
        if self.with_bn:
            for bn in self.bns:
                bn.reset_parameters()

    def fit_with_val(self, features, adj, labels, data, train_iters=200, 
                     initialize=True, verbose=False, 
                     normalize=True, patience=None, noval=False,full=False, **kwargs):
        '''data: full data class'''
        if initialize:
            self.initialize()

        if type(adj) is not torch.Tensor:
            features, adj, labels = utils.to_tensor(features, adj, labels, device=self.device)
        else:
            features = features.to(self.device)
            adj = adj.to(self.device)
            labels = labels.to(self.device)

        if normalize:
            if utils.is_sparse_tensor(adj):
                adj_norm = utils.normalize_adj_tensor(adj, sparse=True)
            else:
                adj_norm = utils.normalize_adj_tensor(adj)   
        else:
            adj_norm = adj

        if 'feat_norm' in kwargs and kwargs['feat_norm']:
            from utils import row_normalize_tensor
            features = row_normalize_tensor(features-features.min())

        self.adj_norm = adj_norm

        self.features = features

        if len(labels.shape) > 1:
            self.multi_label = True
            self.loss = torch.nn.BCELoss()
        else:
            self.multi_label = False
            self.loss = F.nll_loss

        labels = labels.float() if self.multi_label else labels
        self.labels = labels

        if noval:
            self._train_with_val(labels, data, train_iters, verbose, adj_val=True,full=full)
        else:
            self._train_with_val(labels, data, train_iters, verbose,full= full)

    def _train_with_val(self, labels, data, train_iters, verbose, adj_val=False, full= False ):
        if adj_val:
            feat_full, adj_full = data.feat_val, data.adj_val
        else:
            feat_full, adj_full = data.feat_full, data.adj_full
        feat_full, adj_full = utils.to_tensor(feat_full, adj_full, device=self.device)
        adj_full_norm = utils.normalize_adj_tensor(adj_full, sparse=True)
        labels_val = torch.LongTensor(data.labels_val).to(self.device)

        if verbose:
            print('=== training gcn model ===')
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        best_acc_val = 0
        weights = deepcopy(self.state_dict())
        for i in range(train_iters):
            if i == train_iters // 2:
                lr = self.lr*0.1
                optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=self.weight_decay)

            self.train()
            optimizer.zero_grad()
            output = self.forward(self.features, self.adj_norm)
            if full:
                loss_train = self.loss(output[data.idx_train], labels)
            else:
                loss_train = self.loss(output, labels)

            loss_train.backward()
            optimizer.step()

            with torch.no_grad():
                self.eval()
                output = self.forward(feat_full, adj_full_norm)

                if adj_val:
                    loss_val = F.nll_loss(output, labels_val)
                    acc_val = utils.accuracy(output, labels_val)
                else:
                    loss_val = F.nll_loss(output[data.idx_val], labels_val)
                    acc_val = utils.accuracy(output[data.idx_val], labels_val)
                
                if acc_val > best_acc_val:
                    best_acc_val = acc_val
                    self.output = output
                    weights = deepcopy(self.state_dict())
            
        if verbose:
            print('=== picking the best model according to the performance on validation ===')
        self.load_state_dict(weights)

    def test(self, idx_test):
        """Evaluate GCN performance on test set.
        Parameters
        ----------
        idx_test :
            node testing indices
        """
        self.eval()
        output = self.predict()
        # output = self.output
        loss_test = F.nll_loss(output[idx_test], self.labels[idx_test])
        acc_test = utils.accuracy(output[idx_test], self.labels[idx_test])
        print("Test set results:",
              "loss= {:.4f}".format(loss_test.item()),
              "accuracy= {:.4f}".format(acc_test.item()))
        return acc_test.item()

    @torch.no_grad()
    def predict(self, features=None, adj=None, mode='t'):
        """By default, the inputs should be unnormalized adjacency
        Parameters
        ----------
        features :
            node features. If `features` and `adj` are not given, this function will use previous stored `features` and `adj` from training to make predictions.
        adj :
            adjcency matrix. If `features` and `adj` are not given, this function will use previous stored `features` and `adj` from training to make predictions.
        Returns
        -------
        torch.FloatTensor
            output (log probabilities) of GCN
        """

        self.eval()
        if features is None and adj is None:
            return self.forward(self.features, self.adj_norm,mode)
        else:
            if type(adj) is not torch.Tensor:
                features, adj = utils.to_tensor(features, adj, device=self.device)

            self.features = features
            if utils.is_sparse_tensor(adj):
                self.adj_norm = utils.normalize_adj_tensor(adj, sparse=True)
            else:
                self.adj_norm = utils.normalize_adj_tensor(adj)
            return self.forward(self.features, self.adj_norm,mode)

# GAT 
class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

    def forward(self, h, adj):
        adj = adj.to_dense()
        Wh = torch.mm(h, self.W)
        e = self._prepare_attentional_mechanism_input(Wh)

        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, Wh)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])
        e = Wh1 + Wh2.T
        return self.leakyrelu(e)

class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, 
                 nlayers=2, dropout=0.5, alpha=0.2, nheads=8,
                 lr=0.01, weight_decay=5e-4, with_relu=True, with_bias=True, device=None):

        super(GAT, self).__init__()

        assert device is not None, "Please specify 'device'!"
        self.device = device
        self.nfeat = nfeat
        self.nclass = nclass

        self.layers = nn.ModuleList([])
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.with_relu = with_relu
        self.with_bias = with_bias
        self.alpha = alpha
        self.nheads = nheads

        if nlayers == 1:
            self.layers.append(GraphAttentionLayer(nfeat, nclass, dropout=dropout, alpha=alpha, concat=False))
        else:
            self.attentions = nn.ModuleList([GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)])
            for i in range(nlayers-2):
                self.layers.append(GraphAttentionLayer(nhid * nheads, nhid, dropout=dropout, alpha=alpha, concat=True))
            self.layers.append(GraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False))

        self.output = None
        self.best_model = None
        self.best_output = None
        self.adj_norm = None
        self.features = None
        self.multi_label = None

    def forward(self, x, adj, mode='t'):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)

        for i, layer in enumerate(self.layers):
            x = layer(x, adj)
            if self.with_relu:
                x = F.elu(x)

        if mode == 't':
            if self.multi_label:
                return torch.sigmoid(x)
            else:
                return F.log_softmax(x, dim=1)
        else:
            if self.multi_label:
                return torch.sigmoid(x), x
            else:
                return F.log_softmax(x, dim=1), x

    def initialize(self):
        for layer in self.layers:
            layer.reset_parameters()

    # The rest of the methods (fit_with_val, _train_with_val, test, predict) 
    # can remain the same as in the GCN implementation, just replace 'GCN' with 'GAT'
    # where necessary.

    # ... (include the rest of the methods from the GCN class)
    def fit_with_val(self, features, adj, labels, data, train_iters=200, 
                     initialize=True, verbose=False, 
                     normalize=True, patience=None, noval=False,full=False, **kwargs):
        '''data: full data class'''
        if initialize:
            self.initialize()

        if type(adj) is not torch.Tensor:
            features, adj, labels = utils.to_tensor(features, adj, labels, device=self.device)
        else:
            features = features.to(self.device)
            adj = adj.to(self.device)
            labels = labels.to(self.device)

        if normalize:
            if utils.is_sparse_tensor(adj):
                adj_norm = utils.normalize_adj_tensor(adj, sparse=True)
            else:
                adj_norm = utils.normalize_adj_tensor(adj)   
        else:
            adj_norm = adj

        if 'feat_norm' in kwargs and kwargs['feat_norm']:
            from utils import row_normalize_tensor
            features = row_normalize_tensor(features-features.min())

        self.adj_norm = adj_norm

        self.features = features

        if len(labels.shape) > 1:
            self.multi_label = True
            self.loss = torch.nn.BCELoss()
        else:
            self.multi_label = False
            self.loss = F.nll_loss

        labels = labels.float() if self.multi_label else labels
        self.labels = labels

        if noval:
            self._train_with_val(labels, data, train_iters, verbose, adj_val=True,full=full)
        else:
            self._train_with_val(labels, data, train_iters, verbose,full= full)

    def _train_with_val(self, labels, data, train_iters, verbose, adj_val=False, full= False ):
        if adj_val:
            feat_full, adj_full = data.feat_val, data.adj_val
        else:
            feat_full, adj_full = data.feat_full, data.adj_full
        feat_full, adj_full = utils.to_tensor(feat_full, adj_full, device=self.device)
        adj_full_norm = utils.normalize_adj_tensor(adj_full, sparse=True)
        labels_val = torch.LongTensor(data.labels_val).to(self.device)

        if verbose:
            print('=== training gcn model ===')
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        best_acc_val = 0
        weights = deepcopy(self.state_dict())
        for i in range(train_iters):
            if i == train_iters // 2:
                lr = self.lr*0.1
                optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=self.weight_decay)

            self.train()
            optimizer.zero_grad()
            output = self.forward(self.features, self.adj_norm)
            if full:
                loss_train = self.loss(output[data.idx_train], labels)
            else:
                loss_train = self.loss(output, labels)

            loss_train.backward()
            optimizer.step()

            with torch.no_grad():
                self.eval()
                output = self.forward(feat_full, adj_full_norm)

                if adj_val:
                    loss_val = F.nll_loss(output, labels_val)
                    acc_val = utils.accuracy(output, labels_val)
                else:
                    loss_val = F.nll_loss(output[data.idx_val], labels_val)
                    acc_val = utils.accuracy(output[data.idx_val], labels_val)
                
                if acc_val > best_acc_val:
                    best_acc_val = acc_val
                    self.output = output
                    weights = deepcopy(self.state_dict())
            
        if verbose:
            print('=== picking the best model according to the performance on validation ===')
        self.load_state_dict(weights)

    def test(self, idx_test):
        """Evaluate GCN performance on test set.
        Parameters
        ----------
        idx_test :
            node testing indices
        """
        self.eval()
        output = self.predict()
        # output = self.output
        loss_test = F.nll_loss(output[idx_test], self.labels[idx_test])
        acc_test = utils.accuracy(output[idx_test], self.labels[idx_test])
        print("Test set results:",
              "loss= {:.4f}".format(loss_test.item()),
              "accuracy= {:.4f}".format(acc_test.item()))
        return acc_test.item()

    @torch.no_grad()
    def predict(self, features=None, adj=None, mode='t'):
        """By default, the inputs should be unnormalized adjacency
        Parameters
        ----------
        features :
            node features. If `features` and `adj` are not given, this function will use previous stored `features` and `adj` from training to make predictions.
        adj :
            adjcency matrix. If `features` and `adj` are not given, this function will use previous stored `features` and `adj` from training to make predictions.
        Returns
        -------
        torch.FloatTensor
            output (log probabilities) of GCN
        """

        self.eval()
        if features is None and adj is None:
            return self.forward(self.features, self.adj_norm,mode)
        else:
            if type(adj) is not torch.Tensor:
                features, adj = utils.to_tensor(features, adj, device=self.device)

            self.features = features
            if utils.is_sparse_tensor(adj):
                self.adj_norm = utils.normalize_adj_tensor(adj, sparse=True)
            else:
                self.adj_norm = utils.normalize_adj_tensor(adj)
            return self.forward(self.features, self.adj_norm,mode)


# GCNII 
class GCNIIConv(nn.Module):
    def __init__(self, in_features, out_features, alpha, beta):
        super(GCNIIConv, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.beta = beta
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x, adj, h0):
        support = (1 - self.alpha) *x + self.alpha *h0
        output = torch.spmm(adj, support)
        output = (1 - self.beta) *output + self.beta *torch.mm(support, self.weight)
        return output
    

class GCNII(nn.Module):
    def __init__(self, nfeat, nhid, nclass, nlayers=2, dropout=0.5, 
                 lr=0.01, weight_decay=5e-4, alpha=0.1, beta=0.5,
                 with_relu=True, with_bias=False, device=None):

        super(GCNII, self).__init__()

        assert device is not None, "Please specify 'device'!"
        self.device = device
        self.nfeat = nfeat
        self.nclass = nclass
        self.nhid = nhid
        self.nlayers = nlayers

        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(nfeat, nhid))
        for _ in range(nlayers):
            self.layers.append(GCNIIConv(nhid, nhid, alpha, beta))
        self.layers.append(nn.Linear(nhid, nclass))

        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.with_relu = with_relu
        self.with_bias = with_bias
        self.alpha = alpha
        self.beta = beta
        self.output = None
        self.best_model = None
        self.best_output = None
        self.adj_norm = None
        self.features = None
        self.multi_label = None

    def forward(self, x, adj, mode='t'):
        
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.layers[0](x))
        h0 = x
        for i, conv in enumerate(self.layers[1:-1]):
            x = F.dropout(x, self.dropout, training=self.training)
            x = F.relu(conv(x, adj, h0))

        x = F.dropout(x, self.dropout, training=self.training)
        x = self.layers[-1](x)

        if mode == 't':
            if self.multi_label:
                return torch.sigmoid(x)
            else:
                return F.log_softmax(x, dim=1)
        else:
            if self.multi_label:
                return torch.sigmoid(x), x
            else:
                return F.log_softmax(x, dim=1), x

    def initialize(self):
        for layer in self.layers:
            layer.reset_parameters()
    def fit_with_val(self, features, adj, labels, data, train_iters=200, 
                     initialize=True, verbose=False, 
                     normalize=True, patience=None, noval=False,full=False, **kwargs):
        '''data: full data class'''
        if initialize:
            self.initialize()

        if type(adj) is not torch.Tensor:
            features, adj, labels = utils.to_tensor(features, adj, labels, device=self.device)
        else:
            features = features.to(self.device)
            adj = adj.to(self.device)
            labels = labels.to(self.device)

        if normalize:
            if utils.is_sparse_tensor(adj):
                adj_norm = utils.normalize_adj_tensor(adj, sparse=True)
            else:
                adj_norm = utils.normalize_adj_tensor(adj)   
        else:
            adj_norm = adj

        if 'feat_norm' in kwargs and kwargs['feat_norm']:
            from utils import row_normalize_tensor
            features = row_normalize_tensor(features-features.min())

        self.adj_norm = adj_norm

        self.features = features

        if len(labels.shape) > 1:
            self.multi_label = True
            self.loss = torch.nn.BCELoss()
        else:
            self.multi_label = False
            self.loss = F.nll_loss

        labels = labels.float() if self.multi_label else labels
        self.labels = labels

        if noval:
            self._train_with_val(labels, data, train_iters, verbose, adj_val=True,full=full)
        else:
            self._train_with_val(labels, data, train_iters, verbose,full= full)

    def _train_with_val(self, labels, data, train_iters, verbose, adj_val=False, full= False ):
        if adj_val:
            feat_full, adj_full = data.feat_val, data.adj_val
        else:
            feat_full, adj_full = data.feat_full, data.adj_full
        feat_full, adj_full = utils.to_tensor(feat_full, adj_full, device=self.device)
        adj_full_norm = utils.normalize_adj_tensor(adj_full, sparse=True)
        labels_val = torch.LongTensor(data.labels_val).to(self.device)

        if verbose:
            print('=== training gcn model ===')
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        best_acc_val = 0
        weights = deepcopy(self.state_dict())
        for i in range(train_iters):
            if i == train_iters // 2:
                lr = self.lr*0.1
                optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=self.weight_decay)

            self.train()
            optimizer.zero_grad()
            output = self.forward(self.features, self.adj_norm)
            if full:
                loss_train = self.loss(output[data.idx_train], labels)
            else:
                loss_train = self.loss(output, labels)

            loss_train.backward()
            optimizer.step()

            with torch.no_grad():
                self.eval()
                output = self.forward(feat_full, adj_full_norm)

                if adj_val:
                    loss_val = F.nll_loss(output, labels_val)
                    acc_val = utils.accuracy(output, labels_val)
                else:
                    loss_val = F.nll_loss(output[data.idx_val], labels_val)
                    acc_val = utils.accuracy(output[data.idx_val], labels_val)
                
                if acc_val > best_acc_val:
                    best_acc_val = acc_val
                    self.output = output
                    weights = deepcopy(self.state_dict())
            
        if verbose:
            print('=== picking the best model according to the performance on validation ===')
        self.load_state_dict(weights)

    def test(self, idx_test):
        """Evaluate GCN performance on test set.
        Parameters
        ----------
        idx_test :
            node testing indices
        """
        self.eval()
        output = self.predict()
        # output = self.output
        loss_test = F.nll_loss(output[idx_test], self.labels[idx_test])
        acc_test = utils.accuracy(output[idx_test], self.labels[idx_test])
        print("Test set results:",
              "loss= {:.4f}".format(loss_test.item()),
              "accuracy= {:.4f}".format(acc_test.item()))
        return acc_test.item()

    @torch.no_grad()
    def predict(self, features=None, adj=None, mode='t'):
        """By default, the inputs should be unnormalized adjacency
        Parameters
        ----------
        features :
            node features. If `features` and `adj` are not given, this function will use previous stored `features` and `adj` from training to make predictions.
        adj :
            adjcency matrix. If `features` and `adj` are not given, this function will use previous stored `features` and `adj` from training to make predictions.
        Returns
        -------
        torch.FloatTensor
            output (log probabilities) of GCN
        """

        self.eval()
        if features is None and adj is None:
            return self.forward(self.features, self.adj_norm,mode)
        else:
            if type(adj) is not torch.Tensor:
                features, adj = utils.to_tensor(features, adj, device=self.device)

            self.features = features
            if utils.is_sparse_tensor(adj):
                self.adj_norm = utils.normalize_adj_tensor(adj, sparse=True)
            else:
                self.adj_norm = utils.normalize_adj_tensor(adj)
            return self.forward(self.features, self.adj_norm,mode)

# BerNet
class BernConv(nn.Module):
    def __init__(self, in_features, out_features, K, bias=True):
        super(BernConv, self).__init__()
        self.K = K
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x, adj):
        support = torch.mm(x, self.weight)
        output = support
        for _ in range(self.K):
            support = torch.spmm(adj, support)
            output = output + support
        if self.bias is not None:
            output = output + self.bias
        return output

class BernNet(nn.Module):
    def __init__(self, nfeat, nhid, nclass, nlayers=2, dropout=0.5, 
                 lr=0.01, weight_decay=5e-4, K=3, with_bias=True, device=None):

        super(BernNet, self).__init__()

        assert device is not None, "Please specify 'device'!"
        self.device = device
        self.nfeat = nfeat
        self.nclass = nclass
        self.nhid = nhid
        self.nlayers = nlayers

        self.layers = nn.ModuleList()
        self.layers.append(BernConv(nfeat, nhid, K, bias=with_bias))
        for _ in range(nlayers - 2):
            self.layers.append(BernConv(nhid, nhid, K, bias=with_bias))
        self.layers.append(BernConv(nhid, nclass, K, bias=with_bias))

        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.K = K
        self.output = None
        self.best_model = None
        self.best_output = None
        self.adj_norm = None
        self.features = None
        self.multi_label = None

    def initialize(self):
        for layer in self.layers:
            layer.reset_parameters()

    def forward(self, x, adj, mode='t'):
        for i, layer in enumerate(self.layers):
            x = F.dropout(x, self.dropout, training=self.training)
            x = layer(x, adj)
            if i != len(self.layers) - 1:
                x = F.relu(x)
        
        if mode == 't':
            if self.multi_label:
                return torch.sigmoid(x)
            else:
                return F.log_softmax(x, dim=1)
        else:
            if self.multi_label:
                return torch.sigmoid(x), x
            else:
                return F.log_softmax(x, dim=1), x
    
    def fit_with_val(self, features, adj, labels, data, train_iters=200, 
                     initialize=True, verbose=False, 
                     normalize=True, patience=None, noval=False,full=False, **kwargs):
        '''data: full data class'''
        if initialize:
            self.initialize()

        if type(adj) is not torch.Tensor:
            features, adj, labels = utils.to_tensor(features, adj, labels, device=self.device)
        else:
            features = features.to(self.device)
            adj = adj.to(self.device)
            labels = labels.to(self.device)

        if normalize:
            if utils.is_sparse_tensor(adj):
                adj_norm = utils.normalize_adj_tensor(adj, sparse=True)
            else:
                adj_norm = utils.normalize_adj_tensor(adj)   
        else:
            adj_norm = adj

        if 'feat_norm' in kwargs and kwargs['feat_norm']:
            from utils import row_normalize_tensor
            features = row_normalize_tensor(features-features.min())

        self.adj_norm = adj_norm

        self.features = features

        if len(labels.shape) > 1:
            self.multi_label = True
            self.loss = torch.nn.BCELoss()
        else:
            self.multi_label = False
            self.loss = F.nll_loss

        labels = labels.float() if self.multi_label else labels
        self.labels = labels

        if noval:
            self._train_with_val(labels, data, train_iters, verbose, adj_val=True,full=full)
        else:
            self._train_with_val(labels, data, train_iters, verbose,full= full)

    def _train_with_val(self, labels, data, train_iters, verbose, adj_val=False, full= False ):
        if adj_val:
            feat_full, adj_full = data.feat_val, data.adj_val
        else:
            feat_full, adj_full = data.feat_full, data.adj_full
        feat_full, adj_full = utils.to_tensor(feat_full, adj_full, device=self.device)
        adj_full_norm = utils.normalize_adj_tensor(adj_full, sparse=True)
        labels_val = torch.LongTensor(data.labels_val).to(self.device)

        if verbose:
            print('=== training gcn model ===')
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        best_acc_val = 0
        weights = deepcopy(self.state_dict())
        for i in range(train_iters):
            if i == train_iters // 2:
                lr = self.lr*0.1
                optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=self.weight_decay)

            self.train()
            optimizer.zero_grad()
            output = self.forward(self.features, self.adj_norm)
            if full:
                loss_train = self.loss(output[data.idx_train], labels)
            else:
                loss_train = self.loss(output, labels)

            loss_train.backward()
            optimizer.step()

            with torch.no_grad():
                self.eval()
                output = self.forward(feat_full, adj_full_norm)

                if adj_val:
                    loss_val = F.nll_loss(output, labels_val)
                    acc_val = utils.accuracy(output, labels_val)
                else:
                    loss_val = F.nll_loss(output[data.idx_val], labels_val)
                    acc_val = utils.accuracy(output[data.idx_val], labels_val)
                
                if acc_val > best_acc_val:
                    best_acc_val = acc_val
                    self.output = output
                    weights = deepcopy(self.state_dict())
            
        if verbose:
            print('=== picking the best model according to the performance on validation ===')
        self.load_state_dict(weights)

    def test(self, idx_test):
        """Evaluate GCN performance on test set.
        Parameters
        ----------
        idx_test :
            node testing indices
        """
        self.eval()
        output = self.predict()
        # output = self.output
        loss_test = F.nll_loss(output[idx_test], self.labels[idx_test])
        acc_test = utils.accuracy(output[idx_test], self.labels[idx_test])
        print("Test set results:",
              "loss= {:.4f}".format(loss_test.item()),
              "accuracy= {:.4f}".format(acc_test.item()))
        return acc_test.item()

    @torch.no_grad()
    def predict(self, features=None, adj=None, mode='t'):
        """By default, the inputs should be unnormalized adjacency
        Parameters
        ----------
        features :
            node features. If `features` and `adj` are not given, this function will use previous stored `features` and `adj` from training to make predictions.
        adj :
            adjcency matrix. If `features` and `adj` are not given, this function will use previous stored `features` and `adj` from training to make predictions.
        Returns
        -------
        torch.FloatTensor
            output (log probabilities) of GCN
        """

        self.eval()
        if features is None and adj is None:
            return self.forward(self.features, self.adj_norm,mode)
        else:
            if type(adj) is not torch.Tensor:
                features, adj = utils.to_tensor(features, adj, device=self.device)

            self.features = features
            if utils.is_sparse_tensor(adj):
                self.adj_norm = utils.normalize_adj_tensor(adj, sparse=True)
            else:
                self.adj_norm = utils.normalize_adj_tensor(adj)
            return self.forward(self.features, self.adj_norm,mode)


class ChebyConv(nn.Module):
    def __init__(self, in_features, out_features, K, bias=True):
        super(ChebyConv, self).__init__()
        self.K = K
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(K, in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x, adj):
        Tx_0 = x
        Tx_1 = torch.spmm(adj, x)
        out = torch.matmul(Tx_0, self.weight[0])
        out = out + torch.matmul(Tx_1, self.weight[1])

        for k in range(2, self.K):
            Tx_2 = 2 * torch.spmm(adj, Tx_1) - Tx_0
            out = out + torch.matmul(Tx_2, self.weight[k])
            Tx_0, Tx_1 = Tx_1, Tx_2

        if self.bias is not None:
            out = out + self.bias
        return out

class ChebyNet(nn.Module):
    def __init__(self, nfeat, nhid, nclass, nlayers=2, dropout=0.5, 
                 lr=0.01, weight_decay=5e-4, K=3, with_bias=True, device=None):

        super(ChebyNet, self).__init__()

        assert device is not None, "Please specify 'device'!"
        self.device = device
        self.nfeat = nfeat
        self.nclass = nclass
        self.nhid = nhid
        self.nlayers = nlayers

        self.layers = nn.ModuleList()
        self.layers.append(ChebyConv(nfeat, nhid, K, bias=with_bias))
        for _ in range(nlayers - 2):
            self.layers.append(ChebyConv(nhid, nhid, K, bias=with_bias))
        self.layers.append(ChebyConv(nhid, nclass, K, bias=with_bias))

        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.K = K
        self.output = None
        self.best_model = None
        self.best_output = None
        self.adj_norm = None
        self.features = None
        self.multi_label = None

    def initialize(self):
        for layer in self.layers:
            layer.reset_parameters()

    def forward(self, x, adj, mode='t'):
        for i, layer in enumerate(self.layers):
            x = F.dropout(x, self.dropout, training=self.training)
            x = layer(x, adj)
            if i != len(self.layers) - 1:
                x = F.relu(x)
        
        if mode == 't':
            if self.multi_label:
                return torch.sigmoid(x)
            else:
                return F.log_softmax(x, dim=1)
        else:
            if self.multi_label:
                return torch.sigmoid(x), x
            else:
                return F.log_softmax(x, dim=1), x

    def fit_with_val(self, features, adj, labels, data, train_iters=200, 
                     initialize=True, verbose=False, 
                     normalize=True, patience=None, noval=False,full=False, **kwargs):
        '''data: full data class'''
        if initialize:
            self.initialize()

        if type(adj) is not torch.Tensor:
            features, adj, labels = utils.to_tensor(features, adj, labels, device=self.device)
        else:
            features = features.to(self.device)
            adj = adj.to(self.device)
            labels = labels.to(self.device)

        if normalize:
            if utils.is_sparse_tensor(adj):
                adj_norm = utils.normalize_adj_tensor(adj, sparse=True)
            else:
                adj_norm = utils.normalize_adj_tensor(adj)   
        else:
            adj_norm = adj

        if 'feat_norm' in kwargs and kwargs['feat_norm']:
            from utils import row_normalize_tensor
            features = row_normalize_tensor(features-features.min())

        self.adj_norm = adj_norm

        self.features = features

        if len(labels.shape) > 1:
            self.multi_label = True
            self.loss = torch.nn.BCELoss()
        else:
            self.multi_label = False
            self.loss = F.nll_loss

        labels = labels.float() if self.multi_label else labels
        self.labels = labels

        if noval:
            self._train_with_val(labels, data, train_iters, verbose, adj_val=True,full=full)
        else:
            self._train_with_val(labels, data, train_iters, verbose,full= full)

    def _train_with_val(self, labels, data, train_iters, verbose, adj_val=False, full= False ):
        if adj_val:
            feat_full, adj_full = data.feat_val, data.adj_val
        else:
            feat_full, adj_full = data.feat_full, data.adj_full
        feat_full, adj_full = utils.to_tensor(feat_full, adj_full, device=self.device)
        adj_full_norm = utils.normalize_adj_tensor(adj_full, sparse=True)
        labels_val = torch.LongTensor(data.labels_val).to(self.device)

        if verbose:
            print('=== training gcn model ===')
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        best_acc_val = 0
        weights = deepcopy(self.state_dict())
        for i in range(train_iters):
            if i == train_iters // 2:
                lr = self.lr*0.1
                optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=self.weight_decay)

            self.train()
            optimizer.zero_grad()
            output = self.forward(self.features, self.adj_norm)
            if full:
                loss_train = self.loss(output[data.idx_train], labels)
            else:
                loss_train = self.loss(output, labels)

            loss_train.backward()
            optimizer.step()

            with torch.no_grad():
                self.eval()
                output = self.forward(feat_full, adj_full_norm)

                if adj_val:
                    loss_val = F.nll_loss(output, labels_val)
                    acc_val = utils.accuracy(output, labels_val)
                else:
                    loss_val = F.nll_loss(output[data.idx_val], labels_val)
                    acc_val = utils.accuracy(output[data.idx_val], labels_val)
                
                if acc_val > best_acc_val:
                    best_acc_val = acc_val
                    self.output = output
                    weights = deepcopy(self.state_dict())
            
        if verbose:
            print('=== picking the best model according to the performance on validation ===')
        self.load_state_dict(weights)

    def test(self, idx_test):
        """Evaluate GCN performance on test set.
        Parameters
        ----------
        idx_test :
            node testing indices
        """
        self.eval()
        output = self.predict()
        # output = self.output
        loss_test = F.nll_loss(output[idx_test], self.labels[idx_test])
        acc_test = utils.accuracy(output[idx_test], self.labels[idx_test])
        print("Test set results:",
              "loss= {:.4f}".format(loss_test.item()),
              "accuracy= {:.4f}".format(acc_test.item()))
        return acc_test.item()

    @torch.no_grad()
    def predict(self, features=None, adj=None, mode='t'):
        """By default, the inputs should be unnormalized adjacency
        Parameters
        ----------
        features :
            node features. If `features` and `adj` are not given, this function will use previous stored `features` and `adj` from training to make predictions.
        adj :
            adjcency matrix. If `features` and `adj` are not given, this function will use previous stored `features` and `adj` from training to make predictions.
        Returns
        -------
        torch.FloatTensor
            output (log probabilities) of GCN
        """

        self.eval()
        if features is None and adj is None:
            return self.forward(self.features, self.adj_norm,mode)
        else:
            if type(adj) is not torch.Tensor:
                features, adj = utils.to_tensor(features, adj, device=self.device)

            self.features = features
            if utils.is_sparse_tensor(adj):
                self.adj_norm = utils.normalize_adj_tensor(adj, sparse=True)
            else:
                self.adj_norm = utils.normalize_adj_tensor(adj)
            return self.forward(self.features, self.adj_norm,mode)

