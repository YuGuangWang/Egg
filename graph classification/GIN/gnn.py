#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 10 12:20:46 2020

@author: Bxin
"""
import torch
import torch.nn as nn

from torch_geometric.nn import MessagePassing, GCNConv, BatchNorm, GraphSizeNorm, global_max_pool, global_add_pool, global_mean_pool, GlobalAttention

from mSVD import mSVD
#from block_diag import flatten_sym_block
#import time

#%% Graph Conv Layer
class GINConv(MessagePassing):
    def __init__(self, input_dim, hid_dim, norm_type, num_conv_layer):
        super(GINConv, self).__init__()
        if norm_type == 'bn':
            normLayer = BatchNorm(hid_dim)
        elif norm_type =='gn':
            normLayer = GraphSizeNorm()
        else:
            raise Exception('invalid normalization type')
            
        self.mlp = nn.Sequential(*[nn.Linear(input_dim,hid_dim)]+\
                                  [normLayer,
                                   nn.ReLU(),
                                   nn.Linear(hid_dim,hid_dim)]*(num_conv_layer-1))
        ### train eps from default = 0
        self.eps = nn.Parameter(torch.Tensor([0]))

    def forward(self, x, edge_index):
        out = self.mlp((1 + self.eps) *x + self.propagate(edge_index, x=x))   
        return out


# %% Graph Pooling Layer 
def grasspool(hid, graph_sizes, pRatio):
    """
    cur_node_embeddings: hidden rep for a single graph g_i
    hid: hidden rep of batch_graph to be transformed
    graph_sizes: a list of individual graph node size
    """   
    graph_sizes = graph_sizes.tolist()
    node_embeddings = torch.split(hid, graph_sizes)
    ### create an autograd-able variable
    batch_graphs = torch.zeros(len(graph_sizes), int(hid.shape[1]*(hid.shape[1]+1)/2)).to(hid.device)
       
    for g_i in range(len(graph_sizes)):
        cur_node_embeddings = node_embeddings[g_i]
        U, S, V = mSVD.apply(cur_node_embeddings.t())
        k = sum(S > pRatio).item()
        subspace_sym = torch.matmul(U[:,:k],U[:,:k].t())
        ### flatten
        idx = torch.triu_indices(subspace_sym.shape[0],subspace_sym.shape[0])
        cur_graph_tri_u = subspace_sym[idx[0],idx[1]]
        batch_graphs[g_i] = cur_graph_tri_u.flatten()
    return batch_graphs


# def grasspool(hid, graph_sizes, pRatio, mask_upper):
#     """
#     block diag version of GrPool
#     cur_node_embeddings: hidden rep for a single graph g_i
#     hid: hidden rep of batch_graph to be transformed
#     graph_sizes: a list of individual graph node size
#     """   
#     graph_sizes = graph_sizes.tolist()
#     split_set = torch.split(hid.t(), graph_sizes, 1)
    
#     node_embeddings = torch.block_diag(*split_set)
#     bsp = node_embeddings.to_sparse()    
#     t0 = time.time()
#     U,S,V = mSVD.apply(bsp)
#     # U,S,V = mSVD.apply(node_embeddings)
#     #U,S,V=torch.svd_lowrank(bsp,300)
#     print('time for mSVD: {}'.format(time.time()-t0))
#     k = sum(S > pRatio).item()
#     U_sub = U[:,:k]
#     split_U = torch.split(U_sub,len(graph_sizes))    
#     blockU = torch.block_diag(*[i[:,i.abs().sum(0)>3.2e-05] for i in split_U])
#     sub_embedding = torch.sparse.mm(blockU.to_sparse(), blockU.T)

#     mask_upper_vec = sub_embedding[mask_upper==1]
#     batch_graphs = mask_upper_vec.reshape(len(graph_sizes),-1)    
#     return batch_graphs


#%% FC Layer
def score_block(input_dim, output_dim, dropout):
    return nn.Sequential(nn.Linear(input_dim, output_dim),
                         nn.BatchNorm1d(output_dim),
                         nn.ReLU(),
                         nn.Dropout(dropout))


#%% Concat: conv+pool+fc Block
class Block(nn.Module):
    """
    output: a vector of score values for indiviual graphs
    """
    def __init__(self, 
                 num_node_features, num_classes,
                 conv_type, norm_type, conv_hid, num_conv_layer, 
                 pool_type, dropout_ratio, pRatio,
                 hid_fc_dim):
        super(Block, self).__init__()
        self.conv_type = conv_type
        self.pool_type = pool_type
        self.dropout_ratio = dropout_ratio
        self.pRatio = pRatio
        
        ### conv layer
        if self.conv_type == 'gcn':
            self.conv = GCNConv(num_node_features, conv_hid)
        else:
            self.conv = GINConv(num_node_features, conv_hid, norm_type, num_conv_layer)

        ### pool layer
        if self.pool_type == 'attention':
            self.GPool1 = GlobalAttention(nn.Linear(conv_hid, 1, bias=False))
        else:
            self.globalPool = {'max': global_max_pool,
                               'avg': global_mean_pool,
                               'sum': global_add_pool}            
            
        ### fc layer
        if self.pool_type == 'gr':
            hidden_dim = [int(conv_hid*(conv_hid+1)/2)]+hid_fc_dim+[num_classes]
            fcList = [score_block(i,j,self.dropout_ratio) for i,j in zip(hidden_dim[:-2], hidden_dim[1:])]
        else:
            hidden_dim = [conv_hid, num_classes]#
            fcList = []
        ### exclude ReLU and dropout in the last layer
        fcList.append(nn.Sequential(nn.Linear(*hidden_dim[-2:]),
                                    nn.BatchNorm1d(hidden_dim[-1])))
        self.fc = nn.Sequential(*fcList) 
        

    def forward(self, input_rep, batch_idx, graph_size, edge_index):#, mask_upper):
        ### conv layer
        if self.conv_type =='gcn':
            hidden_rep = self.conv(input_rep, edge_index) 
        else: 
            hidden_rep = self.conv(input_rep, edge_index) 
            
        ### pool layer
        if self.pool_type == 'gr':
            h_pooled= grasspool(hidden_rep, graph_size, self.pRatio)
            # h_pooled= grasspool(hidden_rep, graph_size, self.pRatio, mask_upper)
        elif self.pool_type == 'attention':
            h_pooled = self.GPool1(hidden_rep, batch_idx)
        else:
            h_pooled = self.globalPool[self.pool_type](hidden_rep, batch_idx)
        
        ### fc layer
        score = self.fc(h_pooled)
        
        return hidden_rep, score


#%% Model: JK style Full GNN
class GNN(nn.Module):
    def __init__(self, 
                 num_block,
                 num_node_features, num_classes, 
                 conv_type, norm_type, conv_hid, num_conv_layer, 
                 pool_type, dropout_ratio, pRatio,
                 hid_fc_dim):
        super(GNN, self).__init__()
        self.num_block = num_block
        self.num_classes = num_classes
        self.conv_hid = conv_hid
        block_input_dim = [num_node_features]+[conv_hid]*(num_block-2)
        self.block = nn.ModuleList([Block(i, num_classes, 
                                          conv_type, norm_type, conv_hid, num_conv_layer, 
                                          pool_type, dropout_ratio, pRatio, hid_fc_dim)
                                    for i in block_input_dim])
        
    def forward(self, batch_graph):
        edge_index = batch_graph.edge_index
        batch_idx = batch_graph.batch # a long tensor of each node belongs to which graph
        _, graph_sizes = torch.unique(batch_idx, return_counts = True) # a tensor of node size in each graph
        num_graph = len(graph_sizes)
        #mask_upper = flatten_sym_block(self.conv_hid, num_graph)
        
        hidden_rep = batch_graph.x
        hidden_rep_list = []
        final_score = 0
        for layer in range(self.num_block-1):
            hidden_rep, score = self.block[layer](hidden_rep, batch_idx, graph_sizes, edge_index)#, mask_upper)
            hidden_rep_list.append(hidden_rep)
            final_score += score
        
        #return final_score
        if self.num_classes == 1:
            return final_score.view(-1)
        else:
            return final_score
