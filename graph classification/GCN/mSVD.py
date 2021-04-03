#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 23 16:00:41 2020

@author: Professor Junbin Gao
@Copyright:  27 August 2020

This SVD BP implementation follows the following paper [2], and it is similar to [1], 
with Lorentian Broadening for numerical stability, which means an approximated BP algorithm. 
Also I list several other related papers

[1] Hai-Jun Liao, Jin-Guo Liu, Lei Wang and Tao Xiang1,4Differentiable Programming 
    Tensor Networks, PHYSICAL REVIEW X9,031041 (2019) https://journals.aps.org/prx/pdf/10.1103/PhysRevX.9.031041

[2] James Townsend, Differentiating the Singular Value Decomposition, manuscript at 
    https://j-towns.github.io/papers/svd-derivative.pdf   Formulas (31)-(32)

[3] M. Giles. Collected matrix derivative results for forward and reverse mode 
    algorithmic differentiation. In Advances in Automatic Differentiation. 
    Springer LNCSE 64, pages 35–44, 2008.

[4] Wei Wang, Zheng Dang, Yinlin Hu, Pascal Fua, and Mathieu Salzmann, 
    Backpropagation-Friendly Eigendecomposition, 33rd Conference on Neural 
    Information Processing Systems (NeurIPS 2019), Vancouver, Canada, 
    https://papers.nips.cc/paper/8579-backpropagation-friendly-eigendecomposition.pdf

[5] C. Ionescu, O. Vantzos, and C. Sminchisescu,Training DeepNetworks with Structured 
    Layers by Matrix Backpropaga-tion, Proc. IEEE Int. Conf. Comput. Vis. 2965 (2015).

[6] M. Seeger, A. Hetzel, Z. Dai, E. Meissner, and N. D.Lawrence, Auto-Differentiating 
    Linear Algebra,arXiv:1710.08717. Autodiff Workshop, NIPS 2017
"""

import torch
from torch.autograd import Function
import time
# import numpy as np


class mSVD(Function):
    @staticmethod
    def forward(ctx, input):
        # This is an implementation for single matrix only.  It can be extended for a batch of matrices which 
        # are stored in the last two dimension:  To Do List
        # shapes = input.shape  # We get the size of the matrix (m,n)        
        
        U, S, V = torch.svd(input)
        # r = int(input.shape[1]/5)
        # U, S, V = torch.svd_lowrank(input,r)
        # We save them for backpropagation
        ctx.save_for_backward(input, U, S, V) 
        
        return U, S, V

    @staticmethod
    @torch.no_grad()
    def backward(ctx, grad_output_U,grad_output_S,grad_output_V):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # the number of inputs of backward should be same as the output of forwards()
        # All the their size should match in the sequence
              
        _input, U, S, V = ctx.saved_tensors  

        _input = _input.to(device)      
        m = U.shape[0]
        k = U.shape[1]
        n = V.shape[0]
        s1 = torch.unsqueeze(S,-1)  # in shape kx1 with k = min(m,n)
        s2 = torch.unsqueeze(S, 0)  # in shape 1xk with k = min(m,n)
        s_sub = s2 - s1     # in shape kxk
        s_add = s2 + s1     # in shape kxk
      
        ###  Version 2 based on [2]
        # For numerical safe guard
        eps = torch.tensor(1e-12).to(device)#.type(torch.float64).to(device)
        
        idx0 = s_sub == 0
        s_sub[idx0] = eps
        idx1 = torch.abs(s_sub) < eps
        s_sub_sign = torch.sign(s_sub)
        s_sub[idx1] = eps * s_sub_sign[idx1]   # We make sure s_sub is anti-symmetric and safe guarded
        
        idx0 = s_add == 0
        s_add[idx0] = eps
        idx1 = torch.abs(s_add) < eps
        s_add[idx1] = eps                      # We make sure s_sub is symmetric and safe guarded
        
        ss = s_sub * s_add
        idx2 = torch.abs(ss) < eps
        ss_sign = torch.sign(ss)
        ss[idx2] = eps*ss_sign[idx2]
        #print(idx2)
        F = 1/ss    #(s_sub * s_add)  #print(F) 
        
        F[torch.eye(k)==1]= 0.0
        idx = S < eps
        S[idx] = eps
        invS = 1/S 
        
        #t0 = time.time()
        FU = F*(torch.matmul(U.T, grad_output_U) - torch.matmul(grad_output_U.T, U))
        IU = torch.matmul((torch.eye(m).to(device) - torch.matmul(U, U.T)), grad_output_U)
        FV = F*(torch.matmul(V.T, grad_output_V) - torch.matmul(grad_output_V.T, V))
        IV = torch.matmul(grad_output_V.T, (torch.eye(n).to(device) - torch.matmul(V,V.T)))
        
        result = torch.matmul((torch.matmul(U, FU)*S + IU*invS), V.T) + torch.matmul(U*grad_output_S, V.T) + torch.matmul(U, torch.unsqueeze(S,-1)*torch.matmul(FV, V.T) + torch.unsqueeze(invS,-1)*IV)
        #print('time backward: {}'.format(time.time()-t0))
        # Returning tensors
        return grad_output_U.new(result)
    


#%%
# if __name__ == "__main__":
#     hid_rep = torch.load('hid.pt') # [1036, 64]
    
#     hid_rep_t = hid_rep.t()
#     graph_sizes = [54, 42, 38, 108, 23, 25, 39, 6, 24, 27, 12, 21, 18, 23, 20, 14, 15, 34, 33, 21, 52, 19, 40, 12, 50, 4, 59, 16, 24, 43, 100, 20]
#     split_set = torch.split(hid_rep_t,graph_sizes,1)
#     split0 = split_set[0]
#     split1 = split_set[1]
#     block = torch.block_diag(split0,split1)
    
#     U, V, S = torch.svd(block)#mSVD.apply(block)

#     new = torch.matmul(torch.matmul(U,torch.diag(S)),V.T)
    
#     print(torch.allclose(block,new,atol=1e-5))
    
    
# #%%%

# hid_rep = torch.load('hid.pt') # [1036, 64]
    
# hid_rep_t = hid_rep.t()
# graph_sizes = [54, 42, 38, 108, 23, 25, 39, 6, 24, 27, 12, 21, 18, 23, 20, 14, 15, 34, 33, 21, 52, 19, 40, 12, 50, 4, 59, 16, 24, 43, 100, 20]
# split_set = torch.split(hid_rep_t,graph_sizes,1)
# split0 = split_set[0]
# split1 = split_set[1]
# block = torch.block_diag(split0,split1)
    
# U, V, S = torch.svd(block)
    
# #%%
# # import numpy as np
# # import seaborn as sns
# # import matplotlib.pylab as plt
# # ax = sns.heatmap(block.detach().numpy())

# ax = sns.heatmap(V.detach().numpy())
# plt.show()
# #%%
# torch.svd(torch.zeros(10,5))










