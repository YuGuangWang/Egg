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


class mSVD(Function):

    @staticmethod
    def forward(ctx, input):
        # This is an implementation for single matrix only.
        # shapes = input.shape  # We get the size of the matrix (m, n)
        
        U, S, V = torch.svd(input)
        # r = int(input.shape[1] / 5)
        # U, S, V = torch.svd_lowrank(input, r)

        # We save them for backpropagation
        ctx.save_for_backward(input, U, S, V) 
        
        return U, S, V

    @staticmethod
    @torch.no_grad()
    def backward(ctx, grad_output_U, grad_output_S, grad_output_V):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # the number of inputs of backward should be same as the output of forwards()
        # All their size should match in the sequence
              
        input, U, S, V = ctx.saved_tensors
        input = input.to(device)      
        U[U < 1e-6] = 0
        U_sp = U.to_sparse()        
        V[V < 1e-6] = 0
        V_sp = V.to_sparse()  
        
        m, k, n = U_sp.shape[0], U_sp.shape[1], V_sp.shape[0]
        s1 = torch.unsqueeze(S, -1)  # in shape k x 1 with k = min(m, n)
        s2 = torch.unsqueeze(S, 0)  # in shape 1 x k with k = min(m, n)
        s_sub = s2 - s1     # in shape k x k
        s_add = s2 + s1     # in shape k x k

        # Version 2 based on [2]
        # For numerical safe guard
        eps = torch.tensor(1e-12).to(device)
        
        s_sub[s_sub == 0] = eps
        s_sub_sign = torch.sign(s_sub)
        # ensure s_sub is anti-symmetric and safe guarded
        s_sub[torch.abs(s_sub) < eps] = eps * s_sub_sign[torch.abs(s_sub) < eps]
        
        s_add[s_add == 0] = eps
        # ensure s_add is symmetric and safe guarded
        s_add[torch.abs(s_add) < eps] = eps                      
        
        ss = s_sub * s_add
        ss_sign = torch.sign(ss)
        ss[torch.abs(ss) < eps] = eps * ss_sign[torch.abs(ss) < eps]
        F = 1 / ss
        
        F[torch.eye(k) == 1] = 0.0
        S[S < eps] = eps
        invS = 1 / S

        utu = torch.sparse.mm(U_sp.t(), grad_output_U)
        FU = F * (utu - utu.t())
        
        IU = torch.matmul((torch.eye(m).to(device) - torch.sparse.mm(U_sp, U.T)), grad_output_U)

        vtv = torch.sparse.mm(V_sp.t(), grad_output_V)
        FV = F * (vtv - vtv.t())
        
        IV = torch.matmul(grad_output_V.T, (torch.eye(n).to(device) - torch.sparse.mm(V_sp, V.T)))

        result = torch.matmul((torch.sparse.mm(U_sp, FU) * S + IU * invS), V.T) + \
                 torch.sparse.mm(V_sp, (U * grad_output_S).t()).t() + \
                 torch.sparse.mm(U_sp, torch.unsqueeze(S, -1) * (torch.sparse.mm(V_sp, FV.T)).t() + torch.unsqueeze(invS, -1) * IV)

        # Returning tensors
        return grad_output_U.new(result)