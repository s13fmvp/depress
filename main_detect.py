# %%  输入为呼吸信号、脉搏波信号、RR间期和经验特征
# modified at 20250424 在main_new_20250411.py的基础上加入了TFC预训练
# modified at 20250526 适应新数据格式
# modified at 20250612 采用整晚训练整晚验证
# modified at 20250624 加入金字塔输出和损失，纳入心跳信号，问卷损失
# modified at 20251028 实现阴性和阳性的分类，也就是从正常人群中检测出阳性样本
import torch
import torch.nn as nn
import time
from torch.utils.data import DataLoader, Dataset, Subset
import pandas as pd
import numpy as np
from random import randint

import math
from einops import rearrange, repeat
import matplotlib.pyplot as plt
# from model.vit_1d import ViT
import os
from scipy.io import loadmat

import gzip


import psutil
import gc
from dpanmodel import DPANnet_Tiny_Transformer
from dpandataset import ZHEdataset, ZHEdataset_Pretrain


# cpu_num = cpu_count()
# cpu_use = 40
# cur_pid = os.getpid()
# os.sched_setaffinity(cur_pid, list(range(cpu_num))[0*cpu_use:1*cpu_use])



import psutil


import argparse
import sys


# 解析命令行参数
parser = argparse.ArgumentParser(description='Depression1028 main_detect')
parser.add_argument('--fold', type=int, default=2, help='Fold number for cross-validation')
parser.add_argument('--cuda', type=int, default=0, help='Which cuda device to use')
args = parser.parse_args()


def get_memory_usage():
    process = psutil.Process()
    mem_info = process.memory_info()
    mem_used_in_MB = mem_info.rss / 1024 / 1024  # 转成MB
    return mem_used_in_MB


from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

def visualize_embeddings(embeddings, label, method='pca', title='Feature Visualization', save_path=None):
    """
    对Batch_size×embedding_dim的特征做降维并可视化

    Args:
        embeddings (torch.Tensor or np.ndarray): shape = (batch_size, embedding_dim)
        method (str): 'pca' 或 'tsne'，降维方法
        title (str): 图的标题
        save_path (str): 保存路径（可选），如果提供就保存，不提供就直接plt.show()
    """

    if isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.detach().cpu().numpy()  # 转成numpy
    
    if isinstance(label, torch.Tensor):
        label = label.detach().cpu().numpy()  # 转成numpy

    if method == 'pca':
        reducer = PCA(n_components=2)
    elif method == 'tsne':
        reducer = TSNE(n_components=2, init='random', random_state=42)
    else:
        raise ValueError("method should be 'pca' or 'tsne'")

    reduced_embeddings = reducer.fit_transform(embeddings)

    color = ['r', 'g', 'b', 'c']

    plt.figure(figsize=(6,6))
    for i in range(3):
        plt.scatter(reduced_embeddings[label==i, 0], reduced_embeddings[label==i, 1], c=color[i], s=10, alpha=0.7)
    plt.title(title)
    plt.xlabel('Dim 1')
    plt.ylabel('Dim 2')
    plt.grid(True)

    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Feature plot saved to {save_path}")
    else:
        plt.show()

class MultiClassFocalLossWithAlpha(nn.Module):
    
    def __init__(self, alpha=[1/6, 1/6, 1/6, 1/6, 1/6, 1/6], gamma=2, reduction='mean'):
        """
        :param alpha: 权重系数列表, 三分类中第0类权重0.2, 第1类权重0.3, 第2类权重0.5
        :param gamma: 困难样本挖掘的gamma
        :param reduction:
        """
        super(MultiClassFocalLossWithAlpha, self).__init__()
        self.alpha = torch.tensor(alpha)
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, pred, target):
        device = pred.device
        alpha = self.alpha[target].to(device)  # 为当前batch内的样本，逐个分配类别权重，shape=(bs), 一维向量
        log_softmax = torch.log_softmax(pred, dim=1) # 对模型裸输出做softmax再取log, shape=(bs, 3)
        logpt = torch.gather(log_softmax, dim=1, index=target.view(-1, 1))  # 取出每个样本在类别标签位置的log_softmax值, shape=(bs, 1)
        logpt = logpt.view(-1)  # 降维，shape=(bs)
        ce_loss = -logpt  # 对log_softmax再取负，就是交叉熵了
        pt = torch.exp(logpt)  #对log_softmax取exp，把log消了，就是每个样本在类别标签位置的softmax值了，shape=(bs)
        focal_loss = alpha * (1 - pt) ** self.gamma * ce_loss  # 根据公式计算focal loss，得到每个样本的loss值，shape=(bs)
        if self.reduction == "mean":
            return torch.mean(focal_loss)
        if self.reduction == "sum":
            return torch.sum(focal_loss)
        return focal_loss

def duration_loss(prediction):
    """时长限制损失"""
    fs = 2.5
    device = prediction.device
    T = torch.tensor([[[20, 10, 10, 10, 10, 50]]]).to(device)
    B = torch.softmax(prediction, dim=2).to(device)
    C = torch.zeros_like(prediction).to(device)  # 期望持续时间（分钟）
    C[:,1:,:] = B[:,:-1,:] * (C[:,:-1,:] + 1/fs) + (1 - B[:,:-1,:]) * 1/fs
    relu = nn.ReLU(inplace=True)
    Loss = relu(T - C[:,:-1,:]) * (1 - B[:,1:,:])
    loss = torch.mean(Loss)
    return loss



# 数据导入
from torch.utils.data import DataLoader, Dataset, Subset






# 余弦退火学习率
class WarmUpCosineAnnealingLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warm_up_epochs, total_epochs, eta_min=0, last_epoch=-1):
        """
        带 Warm-up 的余弦退火学习率调度器

        Args:
            optimizer: 优化器
            warm_up_epochs: Warm-up 阶段的 epoch 数
            total_epochs: 总共的 epoch 数
            eta_min: 最小学习率
            last_epoch: 上一个 epoch 的索引
        """
        self.warm_up_epochs = warm_up_epochs
        self.total_epochs = total_epochs
        self.eta_min = eta_min
        super(WarmUpCosineAnnealingLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warm_up_epochs:
            # Warm-up 阶段，线性增加学习率
            return [base_lr * (self.last_epoch + 1) / self.warm_up_epochs for base_lr in self.base_lrs]
        else:
            # 余弦退火阶段
            cos_epoch = self.last_epoch - self.warm_up_epochs
            cos_total = self.total_epochs - self.warm_up_epochs
            return [
                self.eta_min + (base_lr - self.eta_min) * 0.5 * (1 + math.cos(math.pi * ((cos_epoch / cos_total) % 1)))
                for base_lr in self.base_lrs
            ]




class NTXentLoss_poly(torch.nn.Module):

    def __init__(self, device, batch_size, temperature, use_cosine_similarity):
        super(NTXentLoss_poly, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.device = device
        self.softmax = torch.nn.Softmax(dim=-1)
        self.mask_samples_from_same_repr = self._get_correlated_mask().type(torch.bool)
        self.similarity_function = self._get_similarity_function(use_cosine_similarity)
        self.criterion = torch.nn.CrossEntropyLoss(reduction="sum")

    def _get_similarity_function(self, use_cosine_similarity):
        if use_cosine_similarity:
            self._cosine_similarity = torch.nn.CosineSimilarity(dim=-1)
            return self._cosine_simililarity
        else:
            return self._dot_simililarity

    def _get_correlated_mask(self):
        diag = np.eye(2 * self.batch_size)
        l1 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=-self.batch_size)
        l2 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=self.batch_size)
        mask = torch.from_numpy((diag + l1 + l2))
        mask = (1 - mask).type(torch.bool)
        return mask.to(self.device)

    @staticmethod
    def _dot_simililarity(x, y):
        v = torch.tensordot(x.unsqueeze(1), y.T.unsqueeze(0), dims=2)
        # x shape: (N, 1, C)
        # y shape: (1, C, 2N)
        # v shape: (N, 2N)
        return v

    def _cosine_simililarity(self, x, y):
        # x shape: (N, 1, C)
        # y shape: (1, 2N, C)
        # v shape: (N, 2N)
        v = self._cosine_similarity(x.unsqueeze(1), y.unsqueeze(0))
        return v

    def forward(self, zis, zjs):
        representations = torch.cat([zjs, zis], dim=0)

        similarity_matrix = self.similarity_function(representations, representations)

        # filter out the scores from the positive samples
        l_pos = torch.diag(similarity_matrix, self.batch_size)
        r_pos = torch.diag(similarity_matrix, -self.batch_size)
        positives = torch.cat([l_pos, r_pos]).view(2 * self.batch_size, 1)

        negatives = similarity_matrix[self.mask_samples_from_same_repr].view(2 * self.batch_size, -1)

        logits = torch.cat((positives, negatives), dim=1)
        logits /= self.temperature

        """Criterion has an internal one-hot function. Here, make all positives as 1 while all negatives as 0. """
        labels = torch.zeros(2 * self.batch_size).to(self.device).long()
        CE = self.criterion(logits, labels)

        onehot_label = torch.cat((torch.ones(2 * self.batch_size, 1),torch.zeros(2 * self.batch_size, negatives.shape[-1])),dim=-1).to(self.device).long()
        # Add poly loss
        pt = torch.mean(onehot_label* torch.nn.functional.softmax(logits,dim=-1))

        epsilon = self.batch_size
        # loss = CE/ (2 * self.batch_size) + epsilon*(1-pt) # replace 1 by 1/self.batch_size
        loss = CE / (2 * self.batch_size) + epsilon * (1/self.batch_size - pt)
        # loss = CE / (2 * self.batch_size)

        return loss

data_dir = '/data108/user_ww/Project/DepressionData/'


cuda_id = args.cuda if hasattr(args, 'cuda') else 2  # 兼容性处理
device = torch.device(f"cuda:{cuda_id}" if torch.cuda.is_available() else "cpu")

mc_flag = False   # 是否使用多标签损失
conver_map = torch.tensor([[3, 1], [0, 2]]).to(device)

# # ************************************************ 预训练模型构建 *********************************************
# pretrain_used = True

# seg_len_pre = 160
# batch_size = 128
# num_pretrain = batch_size * (len(data) // batch_size)
# dataset_pretrain = ZHEdataset_Pretrain(data[:num_pretrain], label[:num_pretrain], seg_len_pre, True, mc_flag, n_disea, fea_data[:num_pretrain], device)
# d_pretrain = DataLoader(dataset_pretrain, batch_size=128, shuffle=True, num_workers=2, pin_memory=True)

# time_encoder = encoder().to(device)
# freq_encoder = encoder().to(device)

# pretrain_epoch = 5000

# params_time = [p for name, p in time_encoder.named_parameters() if p.requires_grad]
# params_freq = [p for name, p in freq_encoder.named_parameters() if p.requires_grad]

# optimizer_pretrain = torch.optim.AdamW([
#     {'params': params_time, 'lr': 0.001},  # 参数组1
#     {'params': params_freq, 'lr': 0.001}  # 参数组2
# ], weight_decay = 1e-4)

# scheduler_pretrain = WarmUpCosineAnnealingLR(optimizer_pretrain, warm_up_epochs=100, total_epochs=200, eta_min=1e-4)

# func_pretrain = NTXentLoss_poly(device=device, batch_size=128, temperature=0.15, use_cosine_similarity=True)

# # ************************************************************************************************************



# ************************************************ 微调模型构建 ***********************************************
finetune_used = True
pretrain_loaded = False

batch_size = 8
seg_len = 7200    # 片段时长（单位：秒）
seg_sample = int(seg_len*2.5)    # 片段采样点数
dynamic = True  # true表示使用整晚数据动态训练
dataset = ZHEdataset(data_dir, seg_len, True, mc_flag, device, dynamic)
dataset_w = ZHEdataset(data_dir, seg_len, False, mc_flag, device, dynamic)



train_index, valid_index, num_list, s_num_list = dataset.split(fold=args.fold, batch_size=batch_size)

d_train = DataLoader(Subset(dataset, train_index), batch_size=batch_size, shuffle=False, num_workers=12, pin_memory=True)
d_train_val = DataLoader(Subset(dataset_w, train_index), batch_size=1, shuffle=False, num_workers=12, pin_memory=True)
d_valid = DataLoader(Subset(dataset_w, valid_index), batch_size=1, shuffle=False, num_workers=12, pin_memory=True)

n_channels = 1
n_classes = 6  # Wake REM N1 N2 N3 离床
n_hidden = 128
n_layers = 2
n_fea = dataset.n_fea
fea_used = True
# model = UNet(n_channels=n_channels, n_classes=n_classes).to(device)
model = DPANnet_Tiny_Transformer(nhid=n_hidden, nlayers=n_layers, nstage=n_classes, ndisea=dataset.n_disea, D=2, flag=mc_flag, nfea=n_fea, fea_used=fea_used).to(device)

# 预训练参数导入
pretrained = False
pretrained_path = "/data108/user_ww/Project/Depression1028/checkpoints/2025-06-08-00-11-46/epoch-844.pth"
# whether to use the pretrained model
if pretrained:
    checkpoint = torch.load(pretrained_path, map_location=device)
    model.load_state_dict(checkpoint["model"], strict=True)

max_epoch = 1100
warm_epoch = 100

name_list = ['conv_dis', 'attn', 'DPANclass']
# params_disea = [p for name, p in model.named_parameters() if p.requires_grad and (
#     name_list[0] in name or name_list[1] in name or name_list[2] in name)]
# params_other = [p for name, p in model.named_parameters() if p.requires_grad and (
#     name_list[0] not in name and name_list[1] not in name and name_list[2] not in name)]
params = [p for name, p in model.named_parameters() if p.requires_grad]


# optimizer = torch.optim.AdamW([
#     {'params': params_other, 'lr': 0.0001},  # 参数组1
#     {'params': params_disea, 'lr': 0.0001}  # 参数组2
# ], weight_decay = 0.05)

# optimizer = torch.optim.SGD(params, lr=0.001, momentum=0.9, weight_decay=0.05)
# optimizer = torch.optim.Adam(params, lr=0.001, weight_decay=0.05)
optimizer = torch.optim.AdamW(params, lr=0.001, weight_decay = 0.05)

scheduler = WarmUpCosineAnnealingLR(optimizer, warm_up_epochs=warm_epoch, total_epochs=400, eta_min=5e-4)

s_num_list
s_weight = (1/n_classes) * torch.sum(torch.tensor(s_num_list)) / torch.tensor(s_num_list)
func = nn.CrossEntropyLoss(weight=s_weight.to(device))
# func = MultiClassFocalLossWithAlpha(reduction='mean')

weight = (1/dataset.n_disea) * torch.sum(torch.tensor(num_list)) / torch.tensor(num_list)
func_dis = nn.CrossEntropyLoss(weight=weight.to(device), reduction='none')

func_score = nn.SmoothL1Loss(reduction='none')
# ***********************************************************************************************************

# 多标签分类时用的损失
if mc_flag:
    num_list_m = [num_list[1]/(num_list[0]+num_list[2]), num_list[0]/(num_list[1]+num_list[2])]
    weight_m = torch.sqrt(torch.tensor(num_list_m))
    # weight_m = (1/2) * torch.sum(torch.tensor(num_list_m)) / torch.sqrt(torch.tensor(num_list_m))
    # print(weight_m)
    func_dis_m = nn.BCEWithLogitsLoss(reduction='mean')

from sklearn.metrics import confusion_matrix

def compute_confusion_matrix(labels, pred_labels_list, gt_labels_list):
    pred_labels_list = np.asarray(pred_labels_list.cpu())
    gt_labels_list = np.asarray(gt_labels_list.cpu())
    matrix = confusion_matrix(gt_labels_list, pred_labels_list, labels=labels)
    return matrix

def compute_kappa(matrix):
    """
    计算kappa系数
    :param matrix:
    :return:
    """
    p0 = np.trace(matrix) / np.sum(matrix)
    pe = 0
    for i in range(len(matrix)):
        pe += np.sum(matrix[i]) * np.sum(matrix[:, i])
    pe = pe / np.sum(matrix) ** 2
    return (p0 - pe) / (1 - pe)


# 训练验证模块
from torch.utils.tensorboard import SummaryWriter

timestamp = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
writer = SummaryWriter(log_dir='/data108/user_ww/Project/Depression1028/logs/{}'.format(timestamp))
# writer = SummaryWriter(log_dir='/data108/user_ww/Project/Depression1028/logs/test')
os.mkdir('/data108/user_ww/Project/Depression1028/checkpoints/{}'.format(timestamp))
os.mkdir('/data108/user_ww/Project/Depression1028/pic/{}'.format(timestamp))


# # **************************************** 自监督学习预训练 ************************************************
# if pretrain_used:

#     for i in range(pretrain_epoch):
#         print("Epoch:{}".format(i))

#         epoch_pretrain_loss_list = []
#         n = 0
        
#         time_encoder.train()
#         freq_encoder.train()


#         for _, (x, r, p, y, x_f, r_f, p_f, aug_x, aug_r, aug_p, aug_x_f, aug_r_f, aug_p_f, zr, mask_attn) in enumerate(d_pretrain):

#             x = x.to(device)
#             r = r.to(device)
#             p = p.to(device)
#             y = y.to(device)
#             x_f = x_f.to(device)
#             r_f = r_f.to(device)
#             p_f = p_f.to(device)
#             aug_x = aug_x.to(device)
#             aug_r = aug_r.to(device)
#             aug_p = aug_p.to(device)
#             aug_x_f = aug_x_f.to(device)
#             aug_r_f = aug_r_f.to(device)
#             aug_p_f = aug_p_f.to(device)        
#             zr = zr.to(device)
#             mask_attn = mask_attn.to(device)

#             optimizer_pretrain.zero_grad()
#             h_x, h_r, h_p, z_x, z_r, z_p = time_encoder(x, r, p)
#             h_x_f, h_r_f, h_p_f, z_x_f, z_r_f, z_p_f = freq_encoder(x_f, r_f, p_f)
#             h_aug_x, h_aug_r, h_aug_p, z_aug_x, z_aug_r, z_aug_p = time_encoder(aug_x, aug_r, aug_p)
#             h_aug_x_f, h_aug_r_f, h_aug_p_f, z_aug_x_f, z_aug_r_f, z_aug_p_f = freq_encoder(aug_x_f, aug_r_f, aug_p_f)

#             lam = 0.2
#             # Loss of x
#             loss_t_x = func_pretrain(h_x, h_aug_x)
#             loss_f_x = func_pretrain(h_x_f, h_aug_x_f)
#             l_TF_x = func_pretrain(z_x, z_x_f)

#             l_1_x, l_2_x, l_3_x = func_pretrain(z_x, z_aug_x_f), func_pretrain(z_aug_x, z_x_f), func_pretrain(z_aug_x, z_aug_x_f)
#             loss_c_x = (1 + l_TF_x - l_1_x) + (1 + l_TF_x - l_2_x) + (1 + l_TF_x - l_3_x)

#             loss_x = lam * (loss_t_x + loss_f_x) + (1 - lam) * loss_c_x

#             # Loss of r
#             loss_t_r = func_pretrain(h_r, h_aug_r)
#             loss_f_r = func_pretrain(h_r_f, h_aug_r_f)
#             l_TF_r = func_pretrain(z_r, z_r_f)

#             l_1_r, l_2_r, l_3_r = func_pretrain(z_r, z_aug_r_f), func_pretrain(z_aug_r, z_r_f), func_pretrain(z_aug_r, z_aug_r_f)
#             loss_c_r = (1 + l_TF_r - l_1_r) + (1 + l_TF_r - l_2_r) + (1 + l_TF_r - l_3_r)

#             loss_r = lam * (loss_t_r + loss_f_r) + (1 - lam) * loss_c_r

#             # Loss of p
#             loss_t_p = func_pretrain(h_p, h_aug_p)
#             loss_f_p = func_pretrain(h_p_f, h_aug_p_f)
#             l_TF_p = func_pretrain(z_p, z_p_f)

#             l_1_p, l_2_p, l_3_p = func_pretrain(z_p, z_aug_p_f), func_pretrain(z_aug_p, z_p_f), func_pretrain(z_aug_p, z_aug_p_f)
#             loss_c_p = (1 + l_TF_p - l_1_p) + (1 + l_TF_p - l_2_p) + (1 + l_TF_p - l_3_p)

#             loss_p = lam * (loss_t_p + loss_f_p) + (1 - lam) * loss_c_p

#             loss = loss_x + loss_r + loss_p

#             loss.backward()
#             optimizer_pretrain.step()
            
#             epoch_pretrain_loss_list.append(loss.item())

#             torch.cuda.empty_cache()

#             if ((i % 50) == 0) and (n == 0):
#                 fea_x = z_x
#                 fea_r = z_r
#                 fea_p = z_p
#                 fea_x_f = z_x_f
#                 fea_r_f = z_r_f
#                 fea_p_f = z_p_f
#                 lab = y
#             else:
#                 fea_x = torch.cat([fea_x, z_x], dim=0)
#                 fea_r = torch.cat([fea_r, z_r], dim=0)
#                 fea_p = torch.cat([fea_p, z_p], dim=0)
#                 fea_x_f = torch.cat([fea_x_f, z_x_f], dim=0)
#                 fea_r_f = torch.cat([fea_r_f, z_r_f], dim=0)
#                 fea_p_f = torch.cat([fea_p_f, z_p_f], dim=0)
#                 lab = torch.cat([lab, y], dim=0)

#             # n = n + xh.shape[0]
#             n = n + x.shape[0]
        
#         # scheduler_pretrain.step()
        
#         print(f"CPU Memory Usage: {get_memory_usage():.2f} MB")

#         Pretrain_Loss = sum(epoch_pretrain_loss_list)/len(epoch_pretrain_loss_list)
#         print("Pretrain Loss:{}".format(Pretrain_Loss))

#         writer.add_scalar(tag="loss/pretrain", scalar_value=Pretrain_Loss, global_step=i)

#         if (i % 50) == 0:
#             visualize_embeddings(fea_x, lab, save_path='/data108/user_ww/Project/Depression1028/pic/{}/fea_x_{}.png'.format(timestamp, i))
#             visualize_embeddings(fea_r, lab, save_path='/data108/user_ww/Project/Depression1028/pic/{}/fea_r_{}.png'.format(timestamp, i))
#             visualize_embeddings(fea_p, lab, save_path='/data108/user_ww/Project/Depression1028/pic/{}/fea_p_{}.png'.format(timestamp, i))
#             visualize_embeddings(fea_x_f, lab, save_path='/data108/user_ww/Project/Depression1028/pic/{}/fea_x_f_{}.png'.format(timestamp, i))
#             visualize_embeddings(fea_r_f, lab, save_path='/data108/user_ww/Project/Depression1028/pic/{}/fea_r_f_{}.png'.format(timestamp, i))
#             visualize_embeddings(fea_p_f, lab, save_path='/data108/user_ww/Project/Depression1028/pic/{}/fea_p_f_{}.png'.format(timestamp, i))




# ***************************************** 有监督学习微调 **************************************************
if finetune_used:
    
    # if pretrain_loaded:
    #     model.load_state_dict(time_encoder.state_dict(), strict=False)

    train_loss_list = []
    train_disea_loss_list = []
    train_disea_loss_1_list = []
    train_disea_loss_2_list = []
    train_disea_loss_3_list = []
    train_score_loss_list = []
    train_acc_list = []
    train_disea_acc_list = []
    valid_loss_list = []
    valid_disea_loss_list = []
    valid_disea_loss_1_list = []
    valid_disea_loss_2_list = []
    valid_disea_loss_3_list = []
    valid_score_loss_list = []
    valid_acc_list = []
    valid_disea_acc_list = []
    valid_axis_list = []
    train_val_loss_list = []
    train_val_disea_loss_list = []
    train_val_acc_list = []
    train_val_disea_acc_list = []
    train_val_axis_list = []
    lr_list = []
    num_epoch = max_epoch

    best_disea_acc = 0
    best_disea_acc_bin = 0
    best_disea_kap = 0
    best_disea_kap_bin = 0
    best_disea_loss = 1000

    for i in range(max_epoch):

        # Monitor CPU memory usage
        process = psutil.Process()
        cpu_memory = process.memory_info().rss / 1024 / 1024  # Convert to MB
        print(f"CPU Memory Usage: {cpu_memory:.2f} MB")

        # Monitor GPU memory usage
        gpu_memory_allocated = torch.cuda.memory_allocated(device) / 1024 / 1024  # Convert to MB
        gpu_memory_cached = torch.cuda.memory_reserved(device) / 1024 / 1024  # Convert to MB
        print(f"GPU Memory Allocated: {gpu_memory_allocated:.2f} MB")
        print(f"GPU Memory Cached: {gpu_memory_cached:.2f} MB")

        # Clear unused memory
        gc.collect()  # Clear CPU memory
        torch.cuda.empty_cache()  # Clear GPU cache
        print("Epoch:{}".format(i))

        # if i < 50:   # 前1000个epoch先做分期预训练
        #     alpha = 1
        #     beta = 1
        #     theta = 0
        # else:   # 后面做精神病分型
        #     alpha = 0
        #     beta = 0
        #     theta = 1

        # # 手动调整学习率
        # if i == 500:  # 第1000个epoch后调整学习率
        #     for param_group in optimizer.param_groups:
        #         param_group['lr'] = 0.0005  # 将学习率调整为1e-4
        #     scheduler.base_lrs = [0.0005 for _ in scheduler.base_lrs]  # 确保 scheduler 不会覆盖
        # if i == 1500:  # 第1000个epoch后调整学习率
        #     for param_group in optimizer.param_groups:
        #         param_group['lr'] = 0.0001  # 将学习率调整为1e-4
        #     scheduler.base_lrs = [0.0001 for _ in scheduler.base_lrs]  # 确保 scheduler 不会覆盖

        alpha = 1
        beta = 0
        theta1 = 1
        theta2 = 0
        theta3 = 0
        theta4 = 0
        gamma = 0
        
        epoch_train_loss_list = []
        epoch_train_acc_list = []
        epoch_train_disea_loss_list = []
        epoch_train_disea_loss_1_list = []
        epoch_train_disea_loss_2_list = []
        epoch_train_disea_loss_3_list = []
        epoch_train_score_loss_list = []
        epoch_train_disea_acc_list = []
        epoch_train_disea_acc_bin_list = []

        n = 0
        
        model.train()
        for _, (x, r, p, f, s, y, zr, mask_attn) in enumerate(d_train):

            x = x.to(device)
            r = r.to(device)
            p = p.to(device)
            f = f.to(device)
            s = s.to(device)
            y = y.to(device)
            zr = zr.to(device)
            mask_attn = mask_attn.to(device)

            if model.name == "DPANnet":
                s = s[:,::32]
                s = s[:,::4]
                mask_attn = mask_attn[:,::48]
                mask_attn = mask_attn[:,::32]
                mask_attn = mask_attn[:,::4]


            optimizer.zero_grad()
            output, logit_dis, logit_dis1, logit_dis2, logit_dis3, score_dis = model(x, r, p, f, s, mask_attn)
            b = output.shape[0]
            c = output.shape[1]
            l = output.shape[2]
            # output_mean = torch.mean(output.reshape(b, c, -1, 10), dim=3) # 采样率10Hz  (b, c=6, l)
            if model.name == "DPANnet":  # ResNet作为backbone输出需降采样32个点，但UNet不用
                output_mean = output
            else:
                output_mean = output
            changes = torch.diff(output_mean, dim=2)
            change_loss = torch.mean(changes**2)  # 变化损失

            # durati_loss = duration_loss(output_mean.permute(0, 2, 1))  # 时长损失
            
            output_fla = output_mean.permute(0,2,1).reshape(-1, n_classes)
            s_fla = s.reshape(-1)
            classi_loss = func(output_fla, s_fla.long())   # 分类损失
            _, s_pred = torch.max(output_fla, dim=1)
            acc = torch.sum(s_pred==s_fla)/len(s_fla)

            # score_loss = ((1 - zr).reshape(-1, 1)**2 * mask_qst * func_score(score_dis, y_qst)).mean() output, logit_dis, logit_dis1, logit_dis2, logit_dis3, score_dis
            # if torch.sum(1*(mask_qst>0)) > 0:
            #     score_loss = torch.sum(1 * (mask_qst>0) * func_score(score_dis, y_qst)) / torch.sum(1 * (mask_qst>0))
            # else:
            #     score_loss = torch.tensor(0)

            if mc_flag:
                # print(logit_dis)
                # print(y)
                disea_loss = func_dis_m(logit_dis, y)   # (batch_size, 2)
                # print(disea_loss)
                label_ind = 1 * (torch.sigmoid(logit_dis) > 0.5)
                # If both elements in a row are < 0.5, set the larger one's label to 1
                zero_rows = (label_ind.sum(dim=1) == 0)
                if zero_rows.any():
                    max_indices = torch.argmax(logit_dis[zero_rows], dim=1)
                    label_ind[zero_rows, max_indices] = 1
                
                y_pred = conver_map[label_ind[:,0], label_ind[:,1]]
                y_fla = conver_map[y[:,0].long(), y[:,1].long()]
                disea_acc = torch.sum(y_pred==y_fla)/len(y_fla)
                # 计算二分类准确率（正类为0）
                y_pred_bin = (y_pred == 0).long()
                y_fla_bin = (y_fla == 0).long()
                disea_acc_bin = torch.sum(y_pred_bin == y_fla_bin) / len(y_fla_bin)
            else:    
                y_fla = y.reshape(-1)
                disea_loss = ((1 - zr)**2 * func_dis(logit_dis, y_fla.long())).mean()
                disea_loss_1 = ((1 - zr)**2 * func_dis(logit_dis1, y_fla.long())).mean()
                disea_loss_2 = ((1 - zr)**2 * func_dis(logit_dis2, y_fla.long())).mean()
                disea_loss_3 = ((1 - zr)**2 * func_dis(logit_dis3, y_fla.long())).mean()
                _, y_pred = torch.max(logit_dis, dim=1)
                disea_acc = torch.sum(y_pred==y_fla)/len(y_fla)
                # 计算二分类准确率（正类为0）
                y_pred_bin = (y_pred == 0).long()
                y_fla_bin = (y_fla == 0).long()
                disea_acc_bin = torch.sum(y_pred_bin == y_fla_bin) / len(y_fla_bin)


            loss = alpha * classi_loss + beta * change_loss + theta1 * disea_loss + theta2 * disea_loss_1 + theta3 * disea_loss_2 + theta4 * disea_loss_3

            assert not torch.isnan(loss)

            with torch.no_grad():
                s_pred_det = s_pred.detach().cpu()
                s_fla_det = s_fla.detach().cpu()
                y_pred_det = y_pred.detach().cpu()
                y_fla_det = y_fla.detach().cpu()

            if n == 0:
                Pre = s_pred_det
                Lab = s_fla_det
                Pre_Disea = y_pred_det
                Lab_Disea = y_fla_det
            else:
                Pre = torch.cat([Pre, s_pred_det])
                Lab = torch.cat([Lab, s_fla_det])
                Pre_Disea = torch.cat([Pre_Disea, y_pred_det])
                Lab_Disea = torch.cat([Lab_Disea, y_fla_det])

            loss.backward()
            optimizer.step()
            # print(model.cnn_breath.conv1.weight.grad.abs().mean().item())
            
            epoch_train_loss_list.append(loss.item())
            epoch_train_acc_list.append(acc.item())
            epoch_train_disea_loss_list.append(disea_loss.item())
            epoch_train_disea_loss_1_list.append(disea_loss_1.item())
            epoch_train_disea_loss_2_list.append(disea_loss_2.item())
            epoch_train_disea_loss_3_list.append(disea_loss_3.item())
            # epoch_train_score_loss_list.append(score_loss.item())
            epoch_train_disea_acc_list.append(disea_acc.item())
            epoch_train_disea_acc_bin_list.append(disea_acc_bin.item())

            torch.cuda.empty_cache()

            # n = n + xh.shape[0]
            n = n + x.shape[0]
        
        lr_list.append(scheduler.get_lr()[0])
        scheduler.step()
        
        Train_Mat = compute_confusion_matrix(labels=[0,1,2,3,4,5], pred_labels_list=Pre, gt_labels_list=Lab)
        Train_Kap = compute_kappa(Train_Mat)
        
        Train_Mat_Disea = compute_confusion_matrix(labels=dataset.label_num, pred_labels_list=Pre_Disea, gt_labels_list=Lab_Disea)
        Train_Kap_Disea = compute_kappa(Train_Mat_Disea)

        Pre_Disea_bin = (Pre_Disea == 0).long()
        Lab_Disea_bin = (Lab_Disea == 0).long()
        Train_Mat_Disea_Bin = compute_confusion_matrix(labels=[0,1], pred_labels_list=Pre_Disea_bin, gt_labels_list=Lab_Disea_bin)
        Train_Kap_Disea_Bin = compute_kappa(Train_Mat_Disea_Bin)

        Train_Loss = sum(epoch_train_loss_list)/len(epoch_train_loss_list)
        Train_Acc = sum(epoch_train_acc_list)/len(epoch_train_acc_list)
        Train_Disea_Loss = sum(epoch_train_disea_loss_list)/len(epoch_train_disea_loss_list)
        Train_Disea_Loss_1 = sum(epoch_train_disea_loss_1_list)/len(epoch_train_disea_loss_1_list)
        Train_Disea_Loss_2 = sum(epoch_train_disea_loss_2_list)/len(epoch_train_disea_loss_2_list)
        Train_Disea_Loss_3 = sum(epoch_train_disea_loss_3_list)/len(epoch_train_disea_loss_3_list)
        # Train_Score_Loss = sum(epoch_train_score_loss_list)/len(epoch_train_score_loss_list)
        Train_Disea_Acc = sum(epoch_train_disea_acc_list)/len(epoch_train_disea_acc_list)
        Train_Disea_Acc_Bin = sum(epoch_train_disea_acc_bin_list)/len(epoch_train_disea_acc_bin_list)
        print("Train Loss:{}".format(Train_Loss))
        print("Train Acc:{}".format(Train_Acc))
        print("Train Disea Loss:{}".format(Train_Disea_Loss))
        print("Train Disea Loss 1:{}".format(Train_Disea_Loss_1))
        print("Train Disea Loss 2:{}".format(Train_Disea_Loss_2))
        print("Train Disea Loss 3:{}".format(Train_Disea_Loss_3))
        # print("Train Score Loss:{}".format(Train_Score_Loss))
        print("Train Disea Acc:{}".format(Train_Disea_Acc))
        print("Train Disea Bin Acc:{}".format(Train_Disea_Acc_Bin))
        print("Train Mat:\n", Train_Mat)
        print("Train Mat Disea:\n", Train_Mat_Disea)

        train_loss_list.append(Train_Loss)
        train_acc_list.append(Train_Acc)
        train_disea_loss_list.append(Train_Disea_Loss)
        # train_score_loss_list.append(Train_Score_Loss)
        train_disea_acc_list.append(Train_Disea_Acc)


        writer.add_scalar(tag="lr", scalar_value=scheduler.get_lr()[0], global_step=i)
        writer.add_scalar(tag="loss/train", scalar_value=Train_Loss, global_step=i)
        writer.add_scalar(tag="acc/train", scalar_value=Train_Acc, global_step=i)
        writer.add_scalar(tag="disea_loss/train", scalar_value=Train_Disea_Loss, global_step=i)
        writer.add_scalar(tag="disea_loss_1/train", scalar_value=Train_Disea_Loss_1, global_step=i)
        writer.add_scalar(tag="disea_loss_2/train", scalar_value=Train_Disea_Loss_2, global_step=i)
        writer.add_scalar(tag="disea_loss_3/train", scalar_value=Train_Disea_Loss_3, global_step=i)
        # writer.add_scalar(tag="score_loss/train", scalar_value=Train_Score_Loss, global_step=i)
        writer.add_scalar(tag="disea_acc/train", scalar_value=Train_Disea_Acc, global_step=i)
        writer.add_scalar(tag="disea_acc_bin/train", scalar_value=Train_Disea_Acc_Bin, global_step=i)
        writer.add_scalar(tag="kappa/train", scalar_value=Train_Kap, global_step=i)
        writer.add_scalar(tag="disea_kappa/train", scalar_value=Train_Kap_Disea, global_step=i)
        writer.add_scalar(tag="disea_kappa_bin/train", scalar_value=Train_Kap_Disea_Bin, global_step=i)

        if (i == 0) or ((i+1) % 5 == 0):  # 每20个epoch验证一次（训练集和验证集都完整的在eval状态下跑一遍）
            # 训练集验证跑一遍
            # epoch_train_val_loss_list = []
            # epoch_train_val_acc_list = []
            # epoch_train_val_disea_loss_list = []
            # epoch_train_val_disea_acc_list = []
            # model.eval()
            # n = 0
            # for _, (x, r, p, f, s, y) in enumerate(d_train_val):

            #     x = x.to(device)
            #     r = r.to(device)
            #     p = p.to(device)
            #     f = f.to(device)
            #     s = s.to(device)
            #     y = y.to(device)

            #     output, logit_dis = model(x, r, p, f)
            #     b = output.shape[0]
            #     c = output.shape[1]
            #     l = output.shape[2]
            #     # output_mean = torch.mean(output.reshape(b, c, -1, 10), dim=3) # 采样率10Hz  (b, c=6, l)
            #     if model.name == "DPANnet":  # ResNet作为backbone输出需降采样32个点，但UNet不用
            #         output_mean = output
            #         s = s[:, ::32]
            #     else:
            #         output_mean = output

            #     changes = torch.diff(output_mean, dim=2)
            #     change_loss = torch.mean(changes**2)  # 变化损失

            #     # durati_loss = duration_loss(output_mean.permute(0, 2, 1))  # 时长损失

            #     output_fla = output_mean.permute(0, 2, 1).reshape(-1, n_classes)
            #     s_fla = s.reshape(-1)
            #     classi_loss = func(output_fla, s_fla.long())   # 分类损失
            #     _, s_pred = torch.max(output_fla, dim=1)
            #     acc = torch.sum(s_pred == s_fla)/len(s_fla)

            #     if mc_flag:
            #         disea_loss = func_dis_m(logit_dis, y)   # (batch_size, 2)
            #         label_ind = 1 * (torch.sigmoid(logit_dis) > 0.5)
            #         y_pred = conver_map[label_ind[:, 0], label_ind[:, 1]]
            #         y_fla = conver_map[y[:, 0].long(), y[:, 1].long()]
            #         disea_acc = torch.sum(y_pred == y_fla)/len(y_fla)
            #     else:
            #         y_fla = y.reshape(-1)
            #         disea_loss = func_dis(logit_dis, y_fla.long())
            #         _, y_pred = torch.max(logit_dis, dim=1)
            #         disea_acc = torch.sum(y_pred == y_fla)/len(y_fla)

            #     loss = alpha * classi_loss + beta * change_loss + theta * disea_loss

            #     if n == 0:
            #         Pre = s_pred
            #         Lab = s_fla
            #         Pre_Disea = y_pred
            #         Lab_Disea = y_fla
            #     else:
            #         Pre = torch.hstack([Pre, s_pred])
            #         Lab = torch.hstack([Lab, s_fla])
            #         Pre_Disea = torch.hstack([Pre_Disea, y_pred])
            #         Lab_Disea = torch.hstack([Lab_Disea, y_fla])

            #     epoch_train_val_loss_list.append(loss.item())
            #     epoch_train_val_acc_list.append(acc.item())
            #     epoch_train_val_disea_loss_list.append(disea_loss.item())
            #     epoch_train_val_disea_acc_list.append(disea_acc.item())

            #     n = n + x.shape[0]

            # Train_Val_Mat = compute_confusion_matrix(labels=[0,1,2,3,4,5], pred_labels_list=Pre, gt_labels_list=Lab)
            # Train_Val_Kap = compute_kappa(Train_Val_Mat)
            # if bi_test:
            #     Train_Val_Mat_Disea = compute_confusion_matrix(labels=[0,1], pred_labels_list=Pre_Disea, gt_labels_list=Lab_Disea)
            # else:
            #     Train_Val_Mat_Disea = compute_confusion_matrix(labels=[0,1,2,3], pred_labels_list=Pre_Disea, gt_labels_list=Lab_Disea)
            # Train_Val_Kap_Disea = compute_kappa(Train_Val_Mat_Disea)

            # Train_Val_Loss = sum(epoch_train_val_loss_list)/len(epoch_train_val_loss_list)
            # Train_Val_Acc = sum(epoch_train_val_acc_list)/len(epoch_train_val_acc_list)
            # Train_Val_Disea_Loss = sum(epoch_train_val_disea_loss_list)/len(epoch_train_val_disea_loss_list)
            # Train_Val_Disea_Acc = sum(epoch_train_val_disea_acc_list)/len(epoch_train_val_disea_acc_list)
            
            # print("Train Val Loss:{}".format(Train_Val_Loss))
            # print("Train Val Acc:{}".format(Train_Val_Acc))
            # print("Train Val Disea Loss:{}".format(Train_Val_Disea_Loss))
            # print("Train Val Disea Acc:{}".format(Train_Val_Disea_Acc))
            # print("Train Val Mat:\n", Train_Val_Mat)
            # print("Train Val Mat Disea:\n", Train_Val_Mat_Disea)

            # train_val_loss_list.append(Train_Val_Loss)
            # train_val_acc_list.append(Train_Val_Acc)
            # train_val_disea_loss_list.append(Train_Val_Disea_Loss)
            # train_val_disea_acc_list.append(Train_Val_Disea_Acc)
            # train_val_axis_list.append(i)

            # writer.add_scalar(tag="loss/train_val", scalar_value=Train_Val_Loss, global_step=i)
            # writer.add_scalar(tag="acc/train_val", scalar_value=Train_Val_Acc, global_step=i)
            # writer.add_scalar(tag="disea_loss/train_val", scalar_value=Train_Val_Disea_Loss, global_step=i)
            # writer.add_scalar(tag="disea_acc/train_val", scalar_value=Train_Val_Disea_Acc, global_step=i)
            # writer.add_scalar(tag="kappa/train_val", scalar_value=Train_Val_Kap, global_step=i)
            # writer.add_scalar(tag="disea_kappa/train_val", scalar_value=Train_Val_Kap_Disea, global_step=i)

            # 验证集跑一遍
            epoch_valid_loss_list = []
            epoch_valid_acc_list = []
            epoch_valid_disea_loss_list = []
            epoch_valid_disea_loss_1_list = []
            epoch_valid_disea_loss_2_list = []
            epoch_valid_disea_loss_3_list = []
            epoch_valid_score_loss_list = []
            epoch_valid_disea_acc_list = []
            epoch_valid_disea_acc_bin_list = []
            model.eval()
            n = 0
            for _, (x, r, p, f, s, y, zr, mask_attn) in enumerate(d_valid):

                x = x.to(device)
                r = r.to(device)
                p = p.to(device)
                f = f.to(device)
                s = s.to(device)
                y = y.to(device)
                zr = zr.to(device)
                mask_attn = mask_attn.to(device)

                # 将整晚数据裁剪成多个和训练长度等长的片段，每个片段都预测一下
                # seg_num = x.shape[-1] // seg_sample
                # x = x[:, :, :seg_num*seg_sample].reshape(seg_num, -1, seg_sample)
                # r = r[:, :, :seg_num*seg_sample].reshape(seg_num, -1, seg_sample)
                # s = s[:, :seg_num*seg_sample].reshape(seg_num, seg_sample)
                # p = p[:, :, :seg_num*seg_sample*48].reshape(seg_num, -1, seg_sample*48)
                # mask_attn = mask_attn[:, :seg_num*seg_sample*48].reshape(seg_num, seg_sample*48)
                # f = f.repeat(seg_num, 1)
                # y = y.repeat(seg_num)
                # y_qst = y_qst.repeat(seg_num, 1)
                # mask_qst = mask_qst.repeat(seg_num, 1)

                if model.name == "DPANnet":
                    s = s[:,::32]
                    s = s[:,::4]
                    mask_attn = mask_attn[:,::48]
                    mask_attn = mask_attn[:,::32]
                    mask_attn = mask_attn[:,::4]                


                output, logit_dis, logit_dis1, logit_dis2, logit_dis3, score_dis = model(x, r, p, f, s, mask_attn)
                b = output.shape[0]
                c = output.shape[1]
                l = output.shape[2]
                # output_mean = torch.mean(output.reshape(b, c, -1, 10), dim=3) # 采样率10Hz  (b, c=6, l)
                if model.name == "DPANnet":  # ResNet作为backbone输出需降采样32个点，但UNet不用
                    output_mean = output
                else:
                    output_mean = output

                changes = torch.diff(output_mean, dim=2)
                change_loss = torch.mean(changes**2)  # 变化损失

                # durati_loss = duration_loss(output_mean.permute(0, 2, 1))  # 时长损失

                output_fla = output_mean.permute(0,2,1).reshape(-1, n_classes)
                s_fla = s.reshape(-1)
                classi_loss = func(output_fla, s_fla.long())   # 分类损失
                _, s_pred = torch.max(output_fla, dim=1)
                acc = torch.sum(s_pred==s_fla)/len(s_fla)

                # score_loss = (mask_qst*func_score(score_dis, y_qst)).mean()
                # if torch.sum(1*(mask_qst>0)) > 0:
                #     score_loss = torch.sum(1 * (mask_qst>0) * func_score(score_dis, y_qst)) / torch.sum(1 * (mask_qst>0))
                # else:
                #     score_loss = torch.tensor(0)

                if mc_flag:
                    disea_loss = func_dis_m(logit_dis, y)   # (batch_size, 2)
                    label_ind = 1 * (torch.sigmoid(logit_dis) > 0.5)
                    # If both elements in a row are < 0.5, set the larger one's label to 1
                    zero_rows = (label_ind.sum(dim=1) == 0)
                    if zero_rows.any():
                        max_indices = torch.argmax(logit_dis[zero_rows], dim=1)
                        label_ind[zero_rows, max_indices] = 1
                    
                    y_pred = conver_map[label_ind[:,0], label_ind[:,1]]
                    y_fla = conver_map[y[:,0].long(), y[:,1].long()]
                    disea_acc = torch.sum(y_pred==y_fla)/len(y_fla)
                    # 计算二分类准确率（正类为0）
                    y_pred_bin = (y_pred == 0).long()
                    y_fla_bin = (y_fla == 0).long()
                    disea_acc_bin = torch.sum(y_pred_bin == y_fla_bin) / len(y_fla_bin)
                else:
                    y_fla = y.reshape(-1)
                    disea_loss = ((1 - zr)**2 * func_dis(logit_dis, y_fla.long())).mean()
                    disea_loss_1 = ((1 - zr)**2 * func_dis(logit_dis1, y_fla.long())).mean()
                    disea_loss_2 = ((1 - zr)**2 * func_dis(logit_dis2, y_fla.long())).mean()
                    disea_loss_3 = ((1 - zr)**2 * func_dis(logit_dis3, y_fla.long())).mean()
                    _, y_pred = torch.max(logit_dis, dim=1)
                    disea_acc = torch.sum(y_pred==y_fla)/len(y_fla)
                    # 计算二分类准确率（正类为0）
                    y_pred_bin = (y_pred == 0).long()
                    y_fla_bin = (y_fla == 0).long()
                    disea_acc_bin = torch.sum(y_pred_bin == y_fla_bin) / len(y_fla_bin)

                loss = alpha * classi_loss + beta * change_loss + theta1 * disea_loss + theta2 * disea_loss_1 + theta3 * disea_loss_2 + theta4 * disea_loss_3

                assert not torch.isnan(loss)

                with torch.no_grad():
                    s_pred_det = s_pred.detach().cpu()
                    s_fla_det = s_fla.detach().cpu()
                    y_pred_det = y_pred.detach().cpu()
                    y_fla_det = y_fla.detach().cpu()

                if n == 0:
                    Pre = s_pred_det
                    Lab = s_fla_det
                    Pre_Disea = y_pred_det
                    Lab_Disea = y_fla_det
                else:
                    Pre = torch.cat([Pre, s_pred_det])
                    Lab = torch.cat([Lab, s_fla_det])
                    Pre_Disea = torch.cat([Pre_Disea, y_pred_det])
                    Lab_Disea = torch.cat([Lab_Disea, y_fla_det])

                epoch_valid_loss_list.append(loss.item())
                epoch_valid_acc_list.append(acc.item())
                epoch_valid_disea_loss_list.append(disea_loss.item())
                epoch_valid_disea_loss_1_list.append(disea_loss_1.item())
                epoch_valid_disea_loss_2_list.append(disea_loss_2.item())
                epoch_valid_disea_loss_3_list.append(disea_loss_3.item())
                # epoch_valid_score_loss_list.append(score_loss.item())
                epoch_valid_disea_acc_list.append(disea_acc.item())
                epoch_valid_disea_acc_bin_list.append(disea_acc_bin.item())

                torch.cuda.empty_cache()

                n = n + x.shape[0]

            Valid_Mat = compute_confusion_matrix(labels=[0,1,2,3,4,5], pred_labels_list=Pre, gt_labels_list=Lab)
            Valid_Kap = compute_kappa(Valid_Mat)

            Valid_Mat_Disea = compute_confusion_matrix(labels=dataset.label_num, pred_labels_list=Pre_Disea, gt_labels_list=Lab_Disea)
            Valid_Kap_Disea = compute_kappa(Valid_Mat_Disea)   

            # 计算二分类Kappa
            Pre_Disea_bin = (Pre_Disea == 0).long()
            Lab_Disea_bin = (Lab_Disea == 0).long()
            Valid_Mat_Disea_Bin = compute_confusion_matrix(labels=[0,1], pred_labels_list=Pre_Disea_bin, gt_labels_list=Lab_Disea_bin)
            Valid_Kap_Disea_Bin = compute_kappa(Valid_Mat_Disea_Bin)
            
            Valid_Loss = sum(epoch_valid_loss_list)/len(epoch_valid_loss_list)
            Valid_Acc = sum(epoch_valid_acc_list)/len(epoch_valid_acc_list)
            Valid_Disea_Loss = sum(epoch_valid_disea_loss_list)/len(epoch_valid_disea_loss_list)
            Valid_Disea_Loss_1 = sum(epoch_valid_disea_loss_1_list)/len(epoch_valid_disea_loss_1_list)
            Valid_Disea_Loss_2 = sum(epoch_valid_disea_loss_2_list)/len(epoch_valid_disea_loss_2_list)
            Valid_Disea_Loss_3 = sum(epoch_valid_disea_loss_3_list)/len(epoch_valid_disea_loss_3_list)
            # Valid_Score_Loss = sum(epoch_valid_score_loss_list)/len(epoch_valid_score_loss_list)
            Valid_Disea_Acc = sum(epoch_valid_disea_acc_list)/len(epoch_valid_disea_acc_list)
            Valid_Disea_Acc_Bin = sum(epoch_valid_disea_acc_bin_list)/len(epoch_valid_disea_acc_bin_list)
            print("Valid Loss:{}".format(Valid_Loss))
            print("Valid Acc:{}".format(Valid_Acc))
            print("Valid Disea Loss:{}".format(Valid_Disea_Loss))
            print("Valid Disea Loss 1:{}".format(Valid_Disea_Loss_1))
            print("Valid Disea Loss 2:{}".format(Valid_Disea_Loss_2))            
            print("Valid Disea Loss 3:{}".format(Valid_Disea_Loss_3))
            # print("Valid Score Loss:{}".format(Valid_Score_Loss))
            print("Valid Disea Acc:{}".format(Valid_Disea_Acc))
            print("Valid Disea Acc Bin:{}".format(Valid_Disea_Acc_Bin))
            print("Valid Mat:\n", Valid_Mat)
            print("Valid Mat Disea:\n", Valid_Mat_Disea)

            if Valid_Disea_Acc > best_disea_acc:
                best_disea_acc = Valid_Disea_Acc
                checkpoint = {}
                checkpoint["model"] = model.state_dict()
                checkpoint["epochs"] = i
                torch.save(checkpoint, '/data108/user_ww/Project/Depression1028/checkpoints/{}/best_disea_acc.pth'.format(timestamp))
            
            if Valid_Disea_Acc_Bin > best_disea_acc_bin:
                best_disea_acc_bin = Valid_Disea_Acc_Bin
                checkpoint = {}
                checkpoint["model"] = model.state_dict()
                checkpoint["epochs"] = i
                torch.save(checkpoint, '/data108/user_ww/Project/Depression1028/checkpoints/{}/best_disea_acc_bin.pth'.format(timestamp))
            
            if Valid_Kap_Disea > best_disea_kap:
                best_disea_kap = Valid_Kap_Disea
                checkpoint = {}
                checkpoint["model"] = model.state_dict()
                checkpoint["epochs"] = i
                torch.save(checkpoint, '/data108/user_ww/Project/Depression1028/checkpoints/{}/best_disea_kap.pth'.format(timestamp))
            
            if Valid_Kap_Disea_Bin > best_disea_kap_bin:
                best_disea_kap_bin = Valid_Kap_Disea_Bin
                checkpoint = {}
                checkpoint["model"] = model.state_dict()
                checkpoint["epochs"] = i
                torch.save(checkpoint, '/data108/user_ww/Project/Depression1028/checkpoints/{}/best_disea_kap_bin.pth'.format(timestamp))

            if Valid_Disea_Loss < best_disea_loss:
                best_disea_loss = Valid_Disea_Loss
                checkpoint = {}
                checkpoint["model"] = model.state_dict()
                checkpoint["epochs"] = i
                torch.save(checkpoint, '/data108/user_ww/Project/Depression1028/checkpoints/{}/best_loss.pth'.format(timestamp))
            
            # torch.save(checkpoint, '/data108/user_ww/Project/Depression1028/checkpoints/{}/epoch-{}.pth'.format(timestamp, i))

            valid_loss_list.append(Valid_Loss)
            valid_acc_list.append(Valid_Acc)
            valid_disea_loss_list.append(Valid_Disea_Loss)
            # valid_score_loss_list.append(Valid_Score_Loss)
            valid_disea_acc_list.append(Valid_Disea_Acc)
            valid_axis_list.append(i)

            writer.add_scalar(tag="loss/valid", scalar_value=Valid_Loss, global_step=i)
            writer.add_scalar(tag="acc/valid", scalar_value=Valid_Acc, global_step=i)
            writer.add_scalar(tag="disea_loss/valid", scalar_value=Valid_Disea_Loss, global_step=i)
            writer.add_scalar(tag="disea_loss_1/valid", scalar_value=Valid_Disea_Loss_1, global_step=i)
            writer.add_scalar(tag="disea_loss_2/valid", scalar_value=Valid_Disea_Loss_2, global_step=i)
            writer.add_scalar(tag="disea_loss_3/valid", scalar_value=Valid_Disea_Loss_3, global_step=i)
            # writer.add_scalar(tag="score_loss/valid", scalar_value=Valid_Score_Loss, global_step=i)
            writer.add_scalar(tag="disea_acc/valid", scalar_value=Valid_Disea_Acc, global_step=i)
            writer.add_scalar(tag="disea_acc_bin/valid", scalar_value=Valid_Disea_Acc_Bin, global_step=i)
            writer.add_scalar(tag="kappa/valid", scalar_value=Valid_Kap, global_step=i)
            writer.add_scalar(tag="disea_kappa/valid", scalar_value=Valid_Kap_Disea, global_step=i)
            writer.add_scalar(tag="disea_kappa_bin/valid", scalar_value=Valid_Kap_Disea_Bin, global_step=i)

        if i == max_epoch - 1:
            checkpoint = {}
            checkpoint["model"] = model.state_dict()
            checkpoint["epochs"] = i
            torch.save(checkpoint, '/data108/user_ww/Project/Depression1028/checkpoints/{}/last.pth'.format(timestamp))

