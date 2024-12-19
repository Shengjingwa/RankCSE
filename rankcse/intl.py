import torch
import torch.nn as nn
import random
import numpy as np
from icecream import ic
class intl(nn.Module):
    def __init__(self, axis=1, iterations=4):
        super(intl, self).__init__()
        self.axis = axis
        self.iterations = iterations

    def forward(self, x):
        # x: shape (N, len_m, len_d)
        assert self.axis in (1,2)
        N, len_m, len_d = x.size()

        if self.axis == 1:
            # Compute mean over axis 1
            m = x.mean(dim=1, keepdim=True)  # Shape: (N, 1, len_d)
            xn = x - m  # Zero-mean input
            # Compute covariance matrix
            sigma = torch.bmm(xn.transpose(1, 2), xn) / (len_m - 1)  # Shape: (N, len_d, len_d)
            eye = torch.eye(len_d, device=x.device).unsqueeze(0).expand(N, -1, -1)  # Shape: (N, len_d, len_d)
        else:
            # Compute mean over axis 2
            m = x.mean(dim=2, keepdim=True)  # Shape: (N, len_m, 1)
            xn = x - m  # Zero-mean input
            # Compute covariance matrix
            sigma = torch.bmm(xn, xn.transpose(1, 2)) / (len_d - 1)  # Shape: (N, len_m, len_m)
            eye = torch.eye(len_m, device=x.device).unsqueeze(0).expand(N, -1, -1)  # Shape: (N, len_m, len_m)

        # Whiten the covariance matrix
        matrix = self.whiten_matrix(sigma, eye)  # Shape: (N, D, D)

        if self.axis == 1:
            decorrelated = torch.bmm(xn, matrix)  # Shape: (N, len_m, len_d)
        else:
            decorrelated = torch.bmm(matrix, xn)  # Shape: (N, len_m, len_d)

        return decorrelated  # Shape: (N, len_m, len_d)

    def whiten_matrix(self, sigma, eye):
        # sigma: Shape (N, D, D)
        trace = torch.sum(torch.diagonal(sigma, dim1=-2, dim2=-1), dim=-1, keepdim=True).unsqueeze(-1)  # Shape: (N, 1, 1)
        sigma_norm = sigma / trace

        projection = eye  # Initial projection matrix
        for _ in range(self.iterations):
            projection = 1.5 * projection - 0.5 * torch.bmm(torch.matrix_power(projection, 3), sigma_norm)

        wm = projection / torch.sqrt(trace)
        return wm  # Shape: (N, D, D)


    
class intl_RGP(nn.Module):
    def __init__(self, num_features=768, gm=1,gd=2,shuffle=False,shuffle2=True,axis=1,iterations=4):
        super(intl_RGP, self).__init__()
        self.num_features = num_features
        self.gm = gm
        self.gd = gd
        self.register_buffer('running_mean', torch.zeros(self.num_features.item() if isinstance(self.num_features, torch.Tensor) else self.num_features))
        self.register_buffer('running_covariance', torch.eye(self.num_features))
        
        self.whiten = intl(axis=axis,iterations=iterations)
        
        self.shuffle = shuffle
        self.shuffle2 = shuffle2
        self.axis=axis
       
    def forward(self,x,num_sent=2):
        
        # x=(128,768) 
        x = x.view(x.shape[0] // num_sent, num_sent, x.shape[1]).transpose(1,0)
        #(2,64,768)

        num_sent, m_dim, d_dim = x.shape
        if self.shuffle :
            # # 打乱
            perm = torch.randperm(x.shape[1]).to(x.device)
            x = x [:,perm,:]
            # print(x.shape)
        if self.shuffle2 :    
            perm2 = torch.randperm(x.shape[-1]).to(x.device)
            # print(f"perm2:{perm2}")
            x = x [:,:,perm2]
            # print(x.shape)
        
        

        assert x.shape[1] % self.gm == 0
        len_m = int(x.shape[1] /self.gm)
        assert x.shape[2] % self.gd == 0
        len_d = int(x.shape[-1] / self.gd)

    

        # 将 x 按照 len_m 和 len_d 进行分块
        # x_blocks 的形状为 (batch_size, num_m_blocks, num_d_blocks, len_m, len_d)
        x_blocks = x.unfold(1, len_m, len_m).unfold(2, len_d, len_d)

        # 调整 x_blocks 的形状以适应批处理
        # x_blocks 的形状为 (batch_size * num_m_blocks * num_d_blocks, len_m, len_d)
        x_blocks = x_blocks.contiguous().view(-1, len_m, len_d)

        # 对所有块同时应用 self.whiten
        x_blocks_whitened = self.whiten(x_blocks)
    


        
        # 将结果 reshape 回原始形状
        x = x_blocks_whitened.view(num_sent, self.gm, self.gd, len_m, len_d)
        x = x.permute(0, 1, 3, 2, 4).contiguous().view(num_sent, m_dim, d_dim)


        # sloss = self.trance_loss(x,num_sent)
        if self.shuffle2  :    
            # 恢复
            x = x[:,:,torch.argsort(perm2)]
            # print(x.shape)
        if self.shuffle :  
            x = x[:,torch.argsort(perm),:]
        # 逆操作 1: 交换维度并合并
        # loss = self.trance_loss(x,num_sent)
        x = x.transpose(1,0).reshape(num_sent * x.shape[1], x.shape[2])  
        # print(x.shape)
        
        # return x
        return x
    



# x = torch.randn(128, 768)


# whitening = intl_RGP()

# x = whitening(x)
# ic(x)
# ic(loss)

