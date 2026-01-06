# %%  输入为呼吸信号、脉搏波信号、RR间期和经验特征
# modified at 20250424 在main_new_20250411.py的基础上加入了TFC预训练
# modified at 20250526 适应新数据格式
# modified at 20250612 采用整晚训练整晚验证
# modified at 20250624 加入金字塔输出和损失，纳入心跳信号，问卷损失
# modified at 20251028 实现阴性和阳性的分类，也就是从正常人群中检测出阳性样本

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
import os
import pickle
import random
import gc
from dpandataset import ZHEdataset
from dpanmodel_vis import DPANnet_Tiny_Transformer


# ---- 必要参数与设备设定 ----
device = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")
data_dir = '/data108/user_ww/Project/DepressionData/'
batch_size = 8
seg_len = 7200      # 片段时长（单位：秒）
dynamic = True
mc_flag = False

# 数据集与索引
dataset = ZHEdataset(data_dir, seg_len, True, mc_flag, device, dynamic)
dataset_w = ZHEdataset(data_dir, seg_len, False, mc_flag, device, dynamic)
train_index, valid_index, num_list, s_num_list = dataset.split(fold=0, batch_size=batch_size)
d_valid = DataLoader(Subset(dataset_w, valid_index), batch_size=1, shuffle=False, num_workers=2, pin_memory=True)

n_classes = 6  # Wake REM N1 N2 N3 离床
n_hidden = 128
n_layers = 2
n_fea = 132
model = DPANnet_Tiny_Transformer(nhid=n_hidden, nlayers=n_layers, nstage=n_classes, ndisea=dataset.n_disea, D=2, flag=mc_flag, nfea=n_fea).to(device)

# ---- 加载特定epoch的权重 ----
checkpoint_path = '/data108/user_ww/Project/Depression1028/checkpoints/2025-10-30-00-44-27/epoch-434.pth'
checkpoint = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(checkpoint["model"], strict=True)
model.eval()

print(f"Loaded weights from {checkpoint_path}")


# ---- 网络关键层权重分布可视化 ----

def plot_weight_distribution(param_tensor, title="权重分布", bins=100):
    param_np = param_tensor.detach().cpu().numpy().flatten()
    plt.figure(figsize=(6,4))
    plt.hist(param_np, bins=bins, alpha=0.7, color='dodgerblue')
    plt.xlabel('Weight value')
    plt.ylabel('Count')
    plt.title(title)
    plt.grid(True)
    plt.show()
    plt.close()

# 选择关注的关键层，可以按需修改
# 观察vit_fuse模块的关键层权重
key_layers = {
    "vit_fuse.transformer.layers.0.0.to_qkv.weight": model.vit_fuse.transformer.layers[0][0].to_qkv.weight,
    "vit_fuse.transformer.layers.0.1.net.0.weight": model.vit_fuse.transformer.layers[0][1].net[0].weight
}

for key, param in key_layers.items():
    print(f"可视化: {key}")
    plot_weight_distribution(param, title=key)

print(f"权重分布可视化图片已在notebook中显示。")

# ---- 可选：结合具体数据输入分析vit_fuse注意力分布 ----

# 从验证集取1个batch的输入（整晚数据）
inputs = d_valid.dataset.__getitem__(0)
x, r, p, f, s, y, zr, mask_attn = [xx.to(device).unsqueeze(dim=0) for xx in inputs]


# 前向传播至vit_fuse模块，分析其注意力输出
# 下面 hook vit_fuse.transformer.layers[0][0] (即第1层Attention)的输出

def get_attn_output_hook(outputs_list):
    def fn(module, input, output):
        outputs_list.append(output[1].detach().cpu())
    return fn

vit_attn_outputs = []
vit_attn_module = model.vit_fuse.transformer.layers[0][0]  # vit_fuse transformer第0层注意力模块
hook_handle = vit_attn_module.register_forward_hook(get_attn_output_hook(vit_attn_outputs))

with torch.no_grad():
    # 为获取vit_fuse attention的真实输入，需走模型前向部分代码。一般模型forward会先对(x,r,p,...)编码，提取特征后送入vit_fuse
    # 如只关注vit_fuse的输入，请确保调用model时会走到vit_fuse.transformer的forward
    # 完整推理
    model.eval()
    _ = model(x, r, p, f, s, mask_attn)  

hook_handle.remove()

if vit_attn_outputs:
    attn_np = vit_attn_outputs[0].flatten().numpy()
    plt.figure(figsize=(6,4))
    plt.hist(attn_np, bins=100, alpha=0.7, color='teal')
    plt.title("vit_fuse Attention输出分布（样例输入）")
    plt.xlabel("Output value")
    plt.ylabel("Count")
    plt.grid(True)
    plt.show()
    plt.close()
    print("vit_fuse Attention层输出分布（样例输入）已在notebook中显示。")

# 你可以按需hook vit_fuse的不同层/不同head，更细致可视化
# 总结：
print("模型vit_fuse关键层注意力/激活分布分析完成。")
