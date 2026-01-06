from flash_attn.flash_attn_interface import flash_attn_unpadded_func
import torch.nn as nn
from einops import rearrange, repeat
from model import resnet1d
from model.fusion import AFF_1D
from vit_classifier_vis import ViT

# 带有睡眠分期mask的Attention模块
class Attention_mask(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x, s=None):  # s输入为0的地方表示遮挡掉的部分 具体地可由（stage==i）生成
        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        if s is not None:
            mask = s.unsqueeze(1).unsqueeze(2)  # (batch, 1, 1, seq_len)
            mask = mask.expand(-1, self.heads, dots.shape[-2], -1)  # (batch, heads, seq_len, seq_len)
            dots = dots.masked_fill(mask == 0, float('-inf'))  # 这里假设 0 表示 mask
        
        attn = self.attend(dots)
        attn = torch.where(torch.isnan(attn), torch.zeros_like(attn), attn)  # 针对mask全0的情况
        
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5
        self.dim_head = dim_head

        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim = -1)

        bs = x.size(0)
        seq = x.size(1)

        q = qkv[0].reshape((bs*seq, self.heads, self.dim_head)).to(torch.float16)
        k = qkv[1].reshape((bs*seq, self.heads, self.dim_head)).to(torch.float16)
        v = qkv[2].reshape((bs*seq, self.heads, self.dim_head)).to(torch.float16)

        lengths = torch.full((bs,), fill_value=seq, device=x.device)
        cu_seqlens = torch.zeros((bs+1, ), device=x.device, dtype=torch.int32)
        cu_seqlens[1:] = lengths.cumsum(0)

        out = flash_attn_unpadded_func(q, k, v, cu_seqlens, cu_seqlens, seq, seq, 0, self.scale, False, False)

        out = out.reshape((bs, seq, self.heads*self.dim_head)).float()

        return self.to_out(out)


class DPANnet(nn.Module):
    # DPANnet is designed for Anxiety & Depression classification, consisting of CNN and LSTM initially.
    # Changes and improvements may be made in the future.
    def __init__(self, nhid, nlayers, nstage, ndisea, D, flag, nfea):
        super(DPANnet, self).__init__()
        self.name = 'DPANnet'
        self.bi = D
        self.nhid = nhid
        self.nlayers = nlayers
        self.nstage = nstage
        self.nfea = nfea
        
        if flag:
            self.ndisea = 2    # 若使用多标签损失，则只检测焦虑和抑郁
        else:
            self.ndisea = ndisea

        # breath CNN module
        self.cnn_breath = resnet1d.resnet18_1d_tiny(ori_channels=1)
        self.conv_cat_breath = nn.Conv1d(512, int(self.nhid*self.bi), 1)
        self.lstm_breath = nn.LSTM(input_size=512, hidden_size=nhid, num_layers=nlayers, batch_first=True, bidirectional=(self.bi==2), dropout=0.1)
        self.fusion_cat_breath = AFF_1D(channels=int(self.nhid*self.bi))

        # rr CNN module
        self.cnn_rr = resnet1d.resnet18_1d_tiny(ori_channels=3)
        self.conv_cat_rr = nn.Conv1d(512, int(self.nhid*self.bi), 1)
        self.lstm_rr = nn.LSTM(input_size=512, hidden_size=nhid, num_layers=nlayers, batch_first=True, bidirectional=(self.bi==2), dropout=0.1)
        self.fusion_cat_rr = AFF_1D(channels=int(self.nhid*self.bi))

        # ppg CNN module
        self.down_ppg = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=3, kernel_size=5, stride=2, padding=1),
            nn.BatchNorm1d(num_features=3),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=3, out_channels=16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(num_features=16),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(num_features=32),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(num_features=64),
            nn.ReLU(inplace=True),            
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, stride=3, padding=1),
            nn.BatchNorm1d(num_features=64),
            nn.ReLU(inplace=True)
        )
        self.cnn_ppg = resnet1d.resnet18_1d_tiny(ori_channels=64)
        self.conv_cat_ppg = nn.Conv1d(512, int(self.nhid*self.bi), 1)
        self.lstm_ppg = nn.LSTM(input_size=512, hidden_size=nhid, num_layers=nlayers, batch_first=True, bidirectional=(self.bi==2), dropout=0.1)
        self.fusion_cat_ppg = AFF_1D(channels=int(self.nhid*self.bi))

        # stage classification module  单个传感器分类的话不用*2  如果是两个传感器数据的话需要*2
        self.decoder = nn.Linear(self.nhid*self.bi*3, 256)
        self.linear = nn.Linear(256, 256)
        self.classify = nn.Linear(256, nstage)
        self.relu = nn.LeakyReLU(negative_slope=0.01, inplace=True)

        # depression anxiety classification module
        self.conv_dis = nn.Sequential(
            nn.Conv1d(in_channels=self.nhid*self.bi*3, out_channels=self.nhid*self.bi*3, kernel_size=1, stride=1),
            nn.BatchNorm1d(num_features=self.nhid*self.bi*3),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=self.nhid*self.bi*3, out_channels=self.nhid*self.bi*3, kernel_size=1, stride=1),
            nn.BatchNorm1d(num_features=self.nhid*self.bi*3),
            nn.ReLU(inplace=True)
        )
        self.attn = Attention(dim=self.nhid*self.bi*3, heads=2, dim_head=64, dropout=0.2)
        self.DPANclass = nn.Linear(self.nhid*self.bi*3 + self.nfea, self.ndisea)  # wo fea的话去掉self.nfea
        self.drop = nn.Dropout(p=0.4)

        # depression anxiety classification module for breath or ppg only
        # self.conv_dis = nn.Sequential(
        #     nn.Conv1d(in_channels=self.nhid*self.bi, out_channels=self.nhid*self.bi, kernel_size=1, stride=1),
        #     nn.BatchNorm1d(num_features=self.nhid*self.bi),
        #     nn.ReLU(inplace=True),
        #     nn.Conv1d(in_channels=self.nhid*self.bi, out_channels=self.nhid*self.bi, kernel_size=1, stride=1),
        #     nn.BatchNorm1d(num_features=self.nhid*self.bi),
        #     nn.ReLU(inplace=True)
        # )
        # self.attn = Attention(dim=self.nhid*self.bi, heads=2, dim_head=64, dropout=0.2)
        # self.DPANclass = nn.Linear(self.nhid*self.bi, self.ndisea)
        # self.drop = nn.Dropout(0.4)

        self.init_weights()

    def init_weights(self):
        init_uniform = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-init_uniform, init_uniform)
        self.classify.bias.data.zero_()
        self.classify.weight.data.uniform_(-init_uniform, init_uniform)
        self.linear.bias.data.zero_()
        self.linear.weight.data.uniform_(-init_uniform, init_uniform)
        self.DPANclass.bias.data.zero_()
        self.DPANclass.weight.data.uniform_(-init_uniform, init_uniform)

    # need to be modified
    def forward(self, breath, rr, ppg, fea, stage=None):

        # breath 特征提取
        embeddings_breath = self.cnn_breath(breath)
        x_cat_breath = self.conv_cat_breath(embeddings_breath)

        batch_num_breath = embeddings_breath.shape[0]
        seq_len_breath = embeddings_breath.shape[-1]
        num_feature_breath = embeddings_breath.shape[1]

        embeddings_breath = embeddings_breath.permute(0, 2, 1)  # (batch_size, seq_len, ninput ninput)
        output_breath, _ = self.lstm_breath(embeddings_breath)  # (batch_size, seq_len, noutput=nhid*self.bi)
        output_breath = output_breath.permute(0, 2, 1)  # (batch_size, noutput, seq_len)

        output_breath = self.fusion_cat_breath(output_breath, x_cat_breath)  # (batch_size, noutput, seq_len)

        output_breath = output_breath.permute(0, 2, 1)  # (batch_size, seq_len, noutput)

        # rr 特征提取
        embeddings_rr = self.cnn_rr(rr)
        x_cat_rr = self.conv_cat_rr(embeddings_rr)

        batch_num_rr = embeddings_rr.shape[0]
        seq_len_rr = embeddings_rr.shape[-1]
        num_feature_rr = embeddings_rr.shape[1]

        embeddings_rr = embeddings_rr.permute(0, 2, 1)  # (batch_size, seq_len, ninput ninput)
        output_rr, _ = self.lstm_rr(embeddings_rr)  # (batch_size, seq_len, noutput=nhid*self.bi)
        output_rr = output_rr.permute(0, 2, 1)  # (batch_size, noutput, seq_len)
        
        output_rr = self.fusion_cat_rr(output_rr, x_cat_rr)  # (batch_size, noutput, seq_len)

        output_rr = output_rr.permute(0, 2, 1)  # (batch_size, seq_len, noutput)

        # ppg 特征提取
        embeddings_ppg = self.down_ppg(ppg)
        embeddings_ppg = self.cnn_ppg(embeddings_ppg)
        x_cat_ppg = self.conv_cat_ppg(embeddings_ppg)

        batch_num_ppg = embeddings_ppg.shape[0]
        seq_len_ppg = embeddings_ppg.shape[-1]
        num_feature_ppg = embeddings_ppg.shape[1]

        embeddings_ppg = embeddings_ppg.permute(0, 2, 1)  # (batch_size, seq_len, ninput ninput)
        output_ppg, _ = self.lstm_ppg(embeddings_ppg)  # (batch_size, seq_len, noutput=nhid*self.bi)
        output_ppg = output_ppg.permute(0, 2, 1)  # (batch_size, noutput, seq_len)
        
        output_ppg = self.fusion_cat_ppg(output_ppg, x_cat_ppg)  # (batch_size, noutput, seq_len)

        output_ppg = output_ppg.permute(0, 2, 1)  # (batch_size, seq_len, noutput)

        # fusion
        output = torch.cat([output_breath, output_rr, output_ppg], dim=2)  # 传感器特征拼接 (batch_size, seq_len, 3*noutput)
        # output = torch.cat([output_rr, output_ppg], dim=2)  # 传感器特征拼接 (batch_size, seq_len, 3*noutput)
        # output = output_ppg

        output_attn = torch.mean(self.attn(self.conv_dis(output.permute(0, 2, 1)).permute(0, 2, 1))[0], dim=1)
        # output_attn = torch.mean(self.conv_dis(output.permute(0, 2, 1)).permute(0, 2, 1), dim=1)  without attention
        # l2_norm = torch.norm(output_attn, p=2, dim=-1, keepdim=True)
        # output_attn = output_attn/l2_norm

        output_dis = torch.cat([output_attn, fea], dim=1)
        # output_dis = output_attn
        disea = self.DPANclass(self.drop(self.relu(output_dis)))

        decoded = self.decoder(output)
        decoded = self.relu(decoded)
        decoded = self.linear(decoded)
        decoded = self.relu(decoded)
        decoded = self.classify(decoded)  # (batch_size, seq_len, nstage=6)
        decoded = decoded.permute(0, 2, 1)  # (batch_size, nstage, seq_len)
        return decoded, disea

import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        # 初始化位置编码矩阵
        pe = torch.zeros(max_len, d_model)
        
        # 生成位置索引
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # 计算频率项
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        # 计算位置编码
        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数维度使用 sin
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇数维度使用 cos
        
        # 将位置编码矩阵注册为缓冲区（不参与训练）
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        # x: (batch_size, seq_len, d_model)
        # 将位置编码添加到输入中
        x = x + self.pe[:, :x.size(1)]
        return x


class encoder(nn.Module):
    def __init__(self):
        super(encoder, self).__init__()
        self.name = 'encoder'       

        # breath CNN module
        self.cnn_breath = resnet1d.resnet18_1d_tiny(ori_channels=1)  # output channel: 128
        self.pro_breath = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 128)
        )

        # rr CNN module
        self.cnn_rr = resnet1d.resnet18_1d_tiny(ori_channels=3)  # output channel: 128
        self.pro_rr = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 128)
        )

        # ppg CNN module
        self.down_ppg = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=3, kernel_size=5, stride=2, padding=1),
            nn.BatchNorm1d(num_features=3),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=3, out_channels=16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(num_features=16),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(num_features=32),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(num_features=64),
            nn.ReLU(inplace=True),            
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, stride=3, padding=1),
            nn.BatchNorm1d(num_features=64),
            nn.ReLU(inplace=True)
        )
        self.cnn_ppg = resnet1d.resnet18_1d_tiny(ori_channels=64)  # output channel: 128
        self.pro_ppg = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 128)
        )
    
    def forward(self, x, r, p):

        h_x = self.cnn_breath(x)
        h_r = self.cnn_rr(r)
        h_p = self.cnn_ppg(self.down_ppg(p))

        L = h_x.shape[-1]
        h_x_center = h_x[:,:,int((L+1)/2-1)]
        h_r_center = h_r[:,:,int((L+1)/2-1)]
        h_p_center = h_p[:,:,int((L+1)/2-1)]

        z_x = self.pro_breath(h_x_center)
        z_r = self.pro_rr(h_r_center)
        z_p = self.pro_ppg(h_p_center)
        
        return h_x_center, h_r_center, h_p_center, z_x, z_r, z_p


# # 使用预训练特征提取器的模型结构(10月28号训练版本)
# class DPANnet_Tiny_Transformer(nn.Module):
#     # DPANnet is designed for Anxiety & Depression classification, consisting of CNN and LSTM initially.
#     # Changes and improvements may be made in the future.
#     def __init__(self, nhid, nlayers, nstage, ndisea, D, flag, nfea):
#         super(DPANnet_Tiny_Transformer, self).__init__()
#         self.name = 'DPANnet'
#         self.bi = D                                                                   
#         self.nhid = nhid
#         self.nlayers = nlayers
#         self.nstage = nstage
#         self.nfea = nfea

#         # Vit 参数
#         self.embedding_dim = 384
#         self.heads = 6
#         self.layers = 12
#         self.stride = 6144  # 120Hz的ppg信号到输入ViT前的stride
        
#         if flag:
#             self.ndisea = 2    # 若使用多标签损失，则只检测焦虑和抑郁
#         else:
#             self.ndisea = ndisea

#         # breath CNN module
#         self.down_breath = nn.Sequential(
#             nn.Conv1d(in_channels=1, out_channels=3, kernel_size=5, stride=2, padding=2),
#             nn.BatchNorm1d(num_features=3),
#             nn.ReLU(inplace=True),
#             nn.Conv1d(in_channels=3, out_channels=16, kernel_size=3, stride=2, padding=1),
#             nn.BatchNorm1d(num_features=16),
#             nn.ReLU(inplace=True)
#         )
#         self.cnn_breath = resnet1d.resnet18_1d_tiny(ori_channels=16)  # output channel: 128
#         self.pro_breath = nn.Sequential(
#             nn.Linear(128, 128),
#             nn.ReLU(inplace=True),
#             nn.Linear(128, 128)
#         )
#         # self.conv_cat_breath = nn.Conv1d(512, int(self.nhid*self.bi), 1)
#         # self.lstm_breath = nn.LSTM(input_size=512, hidden_size=nhid, num_layers=nlayers, batch_first=True, bidirectional=(self.bi==2), dropout=0.1)
#         # self.fusion_cat_breath = AFF_1D(channels=int(self.nhid*self.bi))

#         # rr CNN module
#         self.cnn_rr = resnet1d.resnet18_1d_tiny(ori_channels=3)  # output channel: 128
#         self.pro_rr = nn.Sequential(
#             nn.Linear(128, 128),
#             nn.ReLU(inplace=True),
#             nn.Linear(128, 128)
#         )
#         # self.conv_cat_rr = nn.Conv1d(512, int(self.nhid*self.bi), 1)
#         # self.lstm_rr = nn.LSTM(input_size=512, hidden_size=nhid, num_layers=nlayers, batch_first=True, bidirectional=(self.bi==2), dropout=0.1)
#         # self.fusion_cat_rr = AFF_1D(channels=int(self.nhid*self.bi))

#         # ppg CNN module
#         self.down_ppg = nn.Sequential(
#             nn.Conv1d(in_channels=1, out_channels=3, kernel_size=5, stride=2, padding=2),
#             nn.BatchNorm1d(num_features=3),
#             nn.ReLU(inplace=True),
#             nn.Conv1d(in_channels=3, out_channels=16, kernel_size=3, stride=2, padding=1),
#             nn.BatchNorm1d(num_features=16),
#             nn.ReLU(inplace=True),
#             nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1),
#             nn.BatchNorm1d(num_features=32),
#             nn.ReLU(inplace=True),
#             nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1),
#             nn.BatchNorm1d(num_features=64),
#             nn.ReLU(inplace=True),            
#             nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, stride=3, padding=1),
#             nn.BatchNorm1d(num_features=64),
#             nn.ReLU(inplace=True)
#         )
#         self.cnn_ppg = resnet1d.resnet18_1d_tiny(ori_channels=64)  # output channel: 128
#         self.pro_ppg = nn.Sequential(
#             nn.Linear(128, 128),
#             nn.ReLU(inplace=True),
#             nn.Linear(128, 128)
#         )
#         # self.conv_cat_ppg = nn.Conv1d(512, int(self.nhid*self.bi), 1)
#         # self.lstm_ppg = nn.LSTM(input_size=512, hidden_size=nhid, num_layers=nlayers, batch_first=True, bidirectional=(self.bi==2), dropout=0.1)
#         # self.fusion_cat_ppg = AFF_1D(channels=int(self.nhid*self.bi))

#         self.avepool = nn.AvgPool1d(kernel_size=5, stride=4, padding=2)

#         # Transformer-based feature extraction
#         self.vit_fuse = ViT(
#             seq_len = 256,
#             patch_size = 16,
#             num_classes = 1000,
#             dim = self.embedding_dim,
#             depth = self.layers,
#             heads = self.heads,
#             mlp_dim = 768,
#             dropout = 0.1,
#             emb_dropout = 0.1
#         )

#         # stage classification module  单个传感器分类的话不用*2  如果是两个传感器数据的话需要*2
#         self.decoder = nn.Linear(self.embedding_dim, 128)
#         self.linear1 = nn.Linear(128, 128)
#         self.classify = nn.Linear(128, nstage)
#         self.relu = nn.LeakyReLU(negative_slope=0.01, inplace=True)

#         # depression anxiety classification module
#         self.conv_dis = nn.Sequential(
#             nn.Conv1d(in_channels=self.nhid*self.bi*3, out_channels=self.nhid*self.bi*3, kernel_size=1, stride=1),
#             nn.BatchNorm1d(num_features=self.nhid*self.bi*3),
#             nn.ReLU(inplace=True),
#             nn.Conv1d(in_channels=self.nhid*self.bi*3, out_channels=self.nhid*self.bi*3, kernel_size=1, stride=1),
#             nn.BatchNorm1d(num_features=self.nhid*self.bi*3),
#             nn.ReLU(inplace=True)
#         )
#         self.pos = PositionalEncoding(d_model=self.nhid*self.bi*3)
#         self.attn = Attention(dim=self.nhid*self.bi*3, heads=2, dim_head=64, dropout=0.2)
#         self.linear2 = nn.Linear(self.embedding_dim + self.nfea, self.embedding_dim + self.nfea)
#         self.bn = nn.BatchNorm1d(num_features=self.embedding_dim + self.nfea)
#         self.DPANclass = nn.Linear(self.embedding_dim, self.ndisea)  # wo fea的话去掉self.nfea
#         self.DPANregress1 = nn.Linear(self.embedding_dim, 5)  # 先将矩阵的秩压下来
#         self.DPANregress2 = nn.Linear(5, 16)
#         self.drop = nn.Dropout(p=0.5)

#         self.DPANclass_1 = nn.Linear(192, self.ndisea)
#         self.DPANclass_2 = nn.Linear(192, self.ndisea)
#         self.DPANclass_3 = nn.Linear(384, self.ndisea)

#         # depression anxiety classification module for breath or ppg only
#         # self.conv_dis = nn.Sequential(
#         #     nn.Conv1d(in_channels=self.nhid*self.bi, out_channels=self.nhid*self.bi, kernel_size=1, stride=1),
#         #     nn.BatchNorm1d(num_features=self.nhid*self.bi),
#         #     nn.ReLU(inplace=True),
#         #     nn.Conv1d(in_channels=self.nhid*self.bi, out_channels=self.nhid*self.bi, kernel_size=1, stride=1),
#         #     nn.BatchNorm1d(num_features=self.nhid*self.bi),
#         #     nn.ReLU(inplace=True)
#         # )
#         # self.attn = Attention(dim=self.nhid*self.bi, heads=2, dim_head=64, dropout=0.2)
#         # self.DPANclass = nn.Linear(self.nhid*self.bi, self.ndisea)
#         # self.drop = nn.Dropout(0.4)

#         self.init_weights()

#     def init_weights(self):
#         init_uniform = 0.1
#         self.decoder.bias.data.zero_()
#         self.decoder.weight.data.uniform_(-init_uniform, init_uniform)
#         self.classify.bias.data.zero_()
#         self.classify.weight.data.uniform_(-init_uniform, init_uniform)
#         self.linear1.bias.data.zero_()
#         self.linear1.weight.data.uniform_(-init_uniform, init_uniform)
#         self.linear2.bias.data.zero_()
#         self.linear2.weight.data.uniform_(-init_uniform, init_uniform)
#         self.DPANclass.bias.data.zero_()
#         self.DPANclass.weight.data.uniform_(-init_uniform, init_uniform)
#         self.DPANregress1.bias.data.zero_()
#         self.DPANregress1.weight.data.uniform_(-init_uniform, init_uniform)
#         self.DPANregress2.bias.data.zero_()
#         self.DPANregress2.weight.data.uniform_(-init_uniform, init_uniform)

#     # need to be modified
#     def forward(self, breath, rr, ppg, fea, stage=None, mask=None):

#         # 特征提取（1,2,3分别表示只过了1/2/3个BasicBlock后的特征）
#         h_x, h_x1, h_x2, h_x3 = self.cnn_breath(self.down_breath(breath))
#         h_r, h_r1, h_r2, h_r3 = self.cnn_rr(rr)
#         h_p, h_p1, h_p2, h_p3 = self.cnn_ppg(self.down_ppg(ppg))

#         # 多层级特征输出
#         output_1 = torch.mean(torch.cat([h_x1, h_r1, h_p1], dim=1), dim=2)
#         output_2 = torch.mean(torch.cat([h_x2, h_r2, h_p2], dim=1), dim=2)
#         output_3 = torch.mean(torch.cat([h_x3, h_r3, h_p3], dim=1), dim=2)

#         disea_1 = self.DPANclass_1(output_1)
#         disea_2 = self.DPANclass_2(output_2)
#         disea_3 = self.DPANclass_3(output_3)
        
#         embeddings_breath = self.pro_breath(h_x.permute(0,2,1)).permute(0,2,1)
#         embeddings_rr = self.pro_rr(h_r.permute(0,2,1)).permute(0,2,1)
#         embeddings_ppg = self.pro_ppg(h_p.permute(0,2,1)).permute(0,2,1)   # (batch_size, channel, seq_len)

#         # fusion
#         output = self.avepool(torch.cat([embeddings_breath, embeddings_rr, embeddings_ppg], dim=1)).permute(0,2,1)  # 传感器特征拼接 (batch_size, seq_len, 3*channel)
#         # output = self.avepool(torch.cat([output_breath, output_rr, output_ppg], dim=1)).permute(0,2,1)  # 传感器特征拼接 (batch_size, seq_len, 3*noutput)
#         # output = embeddings_ppg
        
#         mask = mask[:, ::self.stride]
#         cls_tokens, ori_tokens = self.vit_fuse(output, mask)  # cls_tokens: (batch_size, dim=384)    ori_tokens: (batch_size, seq_len, dim=384)

#         # output_dis = torch.cat([cls_tokens, fea], dim=1)
#         output_dis = cls_tokens
#         # output_dis = output_attn
#         disea = self.DPANclass(self.drop(self.relu(output_dis)))
#         score = self.DPANregress2(self.relu(self.DPANregress1(output_dis)))

#         decoded = self.decoder(ori_tokens)
#         decoded = self.relu(decoded)
#         decoded = self.linear1(decoded)
#         decoded = self.relu(decoded)
#         decoded = self.classify(decoded)  # (batch_size, seq_len, nstage=6)
#         decoded = decoded.permute(0, 2, 1)  # (batch_size, nstage, seq_len)
#         return decoded, disea, disea_1, disea_2, disea_3, score
    

# # 10月29号修改简化版
# class DPANnet_Tiny_Transformer(nn.Module):
#     # DPANnet is designed for Anxiety & Depression classification, consisting of CNN and LSTM initially.
#     # Changes and improvements may be made in the future.
#     def __init__(self, nhid, nlayers, nstage, ndisea, D, flag, nfea):
#         super(DPANnet_Tiny_Transformer, self).__init__()
#         self.name = 'DPANnet'
#         self.bi = D                                                                   
#         self.nhid = nhid
#         self.nlayers = nlayers
#         self.nstage = nstage
#         self.nfea = nfea

#         # ViT 参数
#         self.embedding_dim = 384
#         self.heads = 6
#         self.layers = 12
        
#         if flag:
#             self.ndisea = 2    # 若使用多标签损失，则只检测焦虑和抑郁
#         else:
#             self.ndisea = ndisea

#         # breath CNN module
#         self.down_breath = nn.Sequential(
#             nn.Conv1d(in_channels=1, out_channels=3, kernel_size=5, stride=2, padding=2),
#             nn.BatchNorm1d(num_features=3),
#             nn.ReLU(inplace=True),
#             nn.Conv1d(in_channels=3, out_channels=16, kernel_size=3, stride=2, padding=1),
#             nn.BatchNorm1d(num_features=16),
#             nn.ReLU(inplace=True)
#         )
#         self.cnn_breath = resnet1d.resnet18_1d_tiny(ori_channels=16)  # output channel: 128
#         self.pro_breath = nn.Sequential(
#             nn.Linear(128, 128),
#             nn.ReLU(inplace=True),
#             nn.Linear(128, 128)
#         )

#         # rr CNN module
#         self.cnn_rr = resnet1d.resnet18_1d_tiny(ori_channels=3)  # output channel: 128
#         self.pro_rr = nn.Sequential(
#             nn.Linear(128, 128),
#             nn.ReLU(inplace=True),
#             nn.Linear(128, 128)
#         )

#         # ppg CNN module
#         self.down_ppg = nn.Sequential(
#             nn.Conv1d(in_channels=1, out_channels=3, kernel_size=5, stride=2, padding=2),
#             nn.BatchNorm1d(num_features=3),
#             nn.ReLU(inplace=True),
#             nn.Conv1d(in_channels=3, out_channels=16, kernel_size=3, stride=2, padding=1),
#             nn.BatchNorm1d(num_features=16),
#             nn.ReLU(inplace=True),
#             nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1),
#             nn.BatchNorm1d(num_features=32),
#             nn.ReLU(inplace=True),
#             nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1),
#             nn.BatchNorm1d(num_features=64),
#             nn.ReLU(inplace=True),            
#             nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, stride=3, padding=1),
#             nn.BatchNorm1d(num_features=64),
#             nn.ReLU(inplace=True)
#         )
#         self.cnn_ppg = resnet1d.resnet18_1d_tiny(ori_channels=64)  # output channel: 128
#         self.pro_ppg = nn.Sequential(
#             nn.Linear(128, 128),
#             nn.ReLU(inplace=True),
#             nn.Linear(128, 128)
#         )

#         self.avepool = nn.AvgPool1d(kernel_size=5, stride=4, padding=2)

#         # 检查 embedding_dim 用于 ViT
#         self.embedding_dissm = self.embedding_dim

#         # Transformer-based feature extraction
#         self.vit_fuse = ViT(
#             seq_len = 256,
#             patch_size = 16,
#             num_classes = 1000,
#             dim = self.embedding_dissm,
#             depth = self.layers,
#             heads = self.heads,
#             mlp_dim = 768,
#             dropout = 0.1,
#             emb_dropout = 0.1
#         )

#         # stage classification module
#         self.decoder = nn.Linear(self.embedding_dim, 128)
#         self.linear1 = nn.Linear(128, 128)
#         self.classify = nn.Linear(128, nstage)
#         self.relu = nn.LeakyReLU(negative_slope=0.01, inplace=True)

#         # depression anxiety classification module
#         self.linear2 = nn.Linear(self.embedding_dim + self.nfea, self.embedding_dim + self.nfea)
#         self.DPANclass = nn.Linear(self.embedding_dim, self.ndisea)  # wo fea的话去掉self.nfea
#         self.DPANregress1 = nn.Linear(self.embedding_dim, 5)  # 先将矩阵的秩压下来
#         self.DPANregress2 = nn.Linear(5, 16)
#         self.drop = nn.Dropout(p=0.5)

#         self.DPANclass_1 = nn.Linear(192, self.ndisea)
#         self.DPANclass_2 = nn.Linear(192, self.ndisea)
#         self.DPANclass_3 = nn.Linear(384, self.ndisea)

#         # 统一初始化方式
#         self.init_weights()

#     def init_weights(self):
#         # 使用 kaiming_normal_ 初始化所有线性层权重，偏置为 0
#         for m in self.modules():
#             if isinstance(m, nn.Linear):
#                 nn.init.kaiming_normal_(m.weight, nonlinearity='leaky_relu')
#                 if m.bias is not None:
#                     nn.init.zeros_(m.bias)
#             elif isinstance(m, nn.Conv1d):
#                 nn.init.kaiming_normal_(m.weight, nonlinearity='leaky_relu')
#                 if m.bias is not None:
#                     nn.init.zeros_(m.bias)
#             elif isinstance(m, nn.BatchNorm1d):
#                 nn.init.ones_(m.weight)
#                 nn.init.zeros_(m.bias)

#     # need to be modified
#     def forward(self, breath, rr, ppg, fea, stage=None, mask=None):

#         # 特征提取（1,2,3分别表示只过了1/2/3个BasicBlock后的特征）
#         h_x, h_x1, h_x2, h_x3 = self.cnn_breath(self.down_breath(breath))
#         h_r, h_r1, h_r2, h_r3 = self.cnn_rr(rr)
#         h_p, h_p1, h_p2, h_p3 = self.cnn_ppg(self.down_ppg(ppg))

#         # 多层级特征输出
#         output_1 = torch.mean(torch.cat([h_x1, h_r1, h_p1], dim=1), dim=2)
#         output_2 = torch.mean(torch.cat([h_x2, h_r2, h_p2], dim=1), dim=2)
#         output_3 = torch.mean(torch.cat([h_x3, h_r3, h_p3], dim=1), dim=2)

#         disea_1 = self.DPANclass_1(output_1)
#         disea_2 = self.DPANclass_2(output_2)
#         disea_3 = self.DPANclass_3(output_3)
        
#         embeddings_breath = self.pro_breath(h_x.permute(0,2,1)).permute(0,2,1)
#         embeddings_rr = self.pro_rr(h_r.permute(0,2,1)).permute(0,2,1)
#         embeddings_ppg = self.pro_ppg(h_p.permute(0,2,1)).permute(0,2,1)   # (batch_size, channel, seq_len)

#         # fusion
#         output = self.avepool(torch.cat([embeddings_breath, embeddings_rr, embeddings_ppg], dim=1)).permute(0,2,1)  # 传感器特征拼接 (batch_size, seq_len, 3*channel)
#         # output = self.avepool(torch.cat([output_breath, output_rr, output_ppg], dim=1)).permute(0,2,1)  # 传感器特征拼接 (batch_size, seq_len, 3*noutput)
#         # output = embeddings_ppg

#         cls_tokens, ori_tokens = self.vit_fuse(output, mask)  # cls_tokens: (batch_size, dim=384)    ori_tokens: (batch_size, seq_len, dim=384)

#         # output_dis = torch.cat([cls_tokens, fea], dim=1)
#         output_dis = cls_tokens
#         # output_dis = output_attn
#         disea = self.DPANclass(self.drop(self.relu(output_dis)))
#         score = self.DPANregress2(self.relu(self.DPANregress1(output_dis)))

#         decoded = self.decoder(ori_tokens)
#         decoded = self.relu(decoded)
#         decoded = self.linear1(decoded)
#         decoded = self.relu(decoded)
#         decoded = self.classify(decoded)  # (batch_size, seq_len, nstage=6)
#         decoded = decoded.permute(0, 2, 1)  # (batch_size, nstage, seq_len)
#         return decoded, disea, disea_1, disea_2, disea_3, score


# Model-V2
class DPANnet_Tiny_Transformer(nn.Module):
    # DPANnet is designed for Anxiety & Depression classification, consisting of CNN and LSTM initially.
    # Changes and improvements may be made in the future.
    def __init__(self, nhid, nlayers, nstage, ndisea, D, flag, nfea, fea_used=False):
        super(DPANnet_Tiny_Transformer, self).__init__()
        self.name = 'DPANnet'
        self.bi = D                                                                   
        self.nhid = nhid
        self.nlayers = nlayers
        self.nstage = nstage
        self.nfea = nfea
        self.fea_used = fea_used

        # CNN 参数
        self.output_channels = 128

        # ViT 参数
        self.embedding_dim = self.output_channels * 3
        self.heads = 6
        self.layers = 4
        
        if flag:
            self.ndisea = 2    # 若使用多标签损失，则只检测焦虑和抑郁
        else:
            self.ndisea = ndisea

        # breath CNN module
        self.down_breath = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=3, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(num_features=3),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=3, out_channels=16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(num_features=16),
            nn.ReLU(inplace=True)
        )
        self.cnn_breath = resnet1d.resnet18_1d_tiny(ori_channels=16, out_channels=self.output_channels)
        self.pro_breath = nn.Sequential(
            nn.Linear(self.output_channels, self.output_channels),
            nn.ReLU(inplace=True),
            nn.Linear(self.output_channels, self.output_channels)
        )

        # rr CNN module
        self.cnn_rr = resnet1d.resnet18_1d_tiny(ori_channels=3, out_channels=self.output_channels)
        self.pro_rr = nn.Sequential(
            nn.Linear(self.output_channels, self.output_channels),
            nn.ReLU(inplace=True),
            nn.Linear(self.output_channels, self.output_channels)
        )

        # ppg CNN module
        self.down_ppg = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=3, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(num_features=3),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=3, out_channels=16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(num_features=16),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(num_features=32),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(num_features=64),
            nn.ReLU(inplace=True),            
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, stride=3, padding=1),
            nn.BatchNorm1d(num_features=64),
            nn.ReLU(inplace=True)
        )
        self.cnn_ppg = resnet1d.resnet18_1d_tiny(ori_channels=64, out_channels=self.output_channels)
        self.pro_ppg = nn.Sequential(
            nn.Linear(self.output_channels, self.output_channels),
            nn.ReLU(inplace=True),
            nn.Linear(self.output_channels, self.output_channels)
        )

        self.avepool = nn.AvgPool1d(kernel_size=5, stride=4, padding=2)

        # Transformer-based feature extraction
        self.vit_fuse = ViT(
            seq_len = 256,
            patch_size = 16,
            num_classes = 1000,
            dim = self.embedding_dim,
            depth = self.layers,
            heads = self.heads,
            mlp_dim = 768,
            dropout = 0.1,
            emb_dropout = 0.1
        )

        # stage classification module
        self.decoder = nn.Linear(self.embedding_dim, 128)
        self.linear1 = nn.Linear(128, 128)
        self.classify = nn.Linear(128, nstage)
        self.relu = nn.LeakyReLU(negative_slope=0.01, inplace=True)

        # depression anxiety classification module
        if self.fea_used:
            self.DPANclass = nn.Linear(self.embedding_dim + self.nfea, self.ndisea)
            self.DPANregress1 = nn.Linear(self.embedding_dim + self.nfea, 5)  # 先将矩阵的秩压下来
        else:
            self.DPANclass = nn.Linear(self.embedding_dim, self.ndisea)  # wo fea的话去掉self.nfea
            self.DPANregress1 = nn.Linear(self.embedding_dim, 5)  # 先将矩阵的秩压下来
        self.DPANregress2 = nn.Linear(5, 16)
        self.drop = nn.Dropout(p=0.2)

        self.DPANclass_1 = nn.Linear(192, self.ndisea)
        self.DPANclass_2 = nn.Linear(192, self.ndisea)
        self.DPANclass_3 = nn.Linear(3*self.output_channels, self.ndisea)

        # 统一初始化方式
        self.init_weights()

    def init_weights(self):
        # 使用 kaiming_normal_ 初始化所有线性层权重，偏置为 0
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    # need to be modified
    def forward(self, breath, rr, ppg, fea, stage=None, mask=None):

        # 特征提取（1,2,3分别表示只过了1/2/3个BasicBlock后的特征）
        h_x, h_x1, h_x2, h_x3 = self.cnn_breath(self.down_breath(breath))
        h_r, h_r1, h_r2, h_r3 = self.cnn_rr(rr)
        h_p, h_p1, h_p2, h_p3 = self.cnn_ppg(self.down_ppg(ppg))

        # 多层级特征输出
        output_1 = torch.mean(torch.cat([h_x1, h_r1, h_p1], dim=1), dim=2)
        output_2 = torch.mean(torch.cat([h_x2, h_r2, h_p2], dim=1), dim=2)
        output_3 = torch.mean(torch.cat([h_x3, h_r3, h_p3], dim=1), dim=2)

        disea_1 = self.DPANclass_1(output_1)
        disea_2 = self.DPANclass_2(output_2)
        disea_3 = self.DPANclass_3(output_3)
        
        embeddings_breath = self.pro_breath(h_x.permute(0,2,1)).permute(0,2,1)
        embeddings_rr = self.pro_rr(h_r.permute(0,2,1)).permute(0,2,1)
        embeddings_ppg = self.pro_ppg(h_p.permute(0,2,1)).permute(0,2,1)   # (batch_size, channel, seq_len)

        # fusion
        output = self.avepool(torch.cat([embeddings_breath, embeddings_rr, embeddings_ppg], dim=1)).permute(0,2,1)  # 传感器特征拼接 (batch_size, seq_len, 3*channel)

        cls_tokens, ori_tokens = self.vit_fuse(output, mask)  # cls_tokens: (batch_size, dim=384)    ori_tokens: (batch_size, seq_len, dim=384)

        if self.fea_used:
            output_dis = torch.cat([cls_tokens, fea], dim=1)
        else:
            output_dis = cls_tokens

        disea = self.DPANclass(self.drop(self.relu(output_dis)))
        score = self.DPANregress2(self.relu(self.DPANregress1(output_dis)))

        decoded = self.decoder(ori_tokens)
        decoded = self.relu(decoded)
        decoded = self.linear1(decoded)
        decoded = self.relu(decoded)
        decoded = self.classify(decoded)  # (batch_size, seq_len, nstage=6)
        decoded = decoded.permute(0, 2, 1)  # (batch_size, nstage, seq_len)
        return decoded, disea, disea_1, disea_2, disea_3, score