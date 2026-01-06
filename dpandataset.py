from scipy.signal import butter, filtfilt
from sklearn.model_selection import StratifiedKFold
from scipy.interpolate import interp1d
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, Subset
import torch
import random
import torch.nn.functional as F
import numpy as np
import pickle
import os
from collections import Counter
import math


# 补零函数
def pad_to_multiple(tensor, stride):
    """
    Pads a tensor along its last dimension (L) so that its size is a multiple of 32.
    
    Args:
        tensor (torch.Tensor): Input tensor of shape (B, C, L).
        
    Returns:
        torch.Tensor: Padded tensor with the same shape for (B, C) and L padded to be a multiple of 32.
    """
    
    L = tensor.shape[-1]
    # Calculate the padding size
    padding_size = (stride - (L % stride)) % stride
    # Pad the tensor along the last dimension
    padded_tensor = torch.nn.functional.pad(tensor, (0, padding_size), mode='constant', value=0)
    return padded_tensor

# 加噪声
def add_noise(signal):
    noise_level = 0.3 * torch.sqrt(torch.var(signal))
    noise = torch.randn_like(signal) * noise_level
    return signal + noise

# 时间缩放
def time_scaling(signal, scale_factor=1.2, mode='linear'):  # 呼吸、脉搏波、分期分别缩放
    scaling_signal = signal.clone()
    length = signal.size(-1)
    scaled_length = int(length * scale_factor)
    if len(signal.shape) == 1:
        scaling_signal = F.interpolate(scaling_signal.unsqueeze(0).unsqueeze(0), size=scaled_length, mode=mode)
    elif len(signal.shape) == 2:
        scaling_signal = F.interpolate(scaling_signal.unsqueeze(0), size=scaled_length, mode=mode)
    return scaling_signal.squeeze()

# 时域遮挡
def time_mask(signal):
    masked_signal = signal.clone()
    length = signal.size(-1)
    mask_length = round(0.25*length)
    start = torch.randint(0, length - mask_length + 1, (1,)).item()
    masked_signal[..., start:start + mask_length] = 0
    return masked_signal

def remove_frequency(x, pertub_ratio=0.0):
    mask = torch.rand(x.shape, device=x.device) > pertub_ratio
    mask = mask.to(x.device)
    return x*mask

def add_frequency(x, pertub_ratio=0.0):
    mask = torch.rand(x.shape, device=x.device) > (1 - pertub_ratio)
    mask = mask.to(x.device)
    max_amplitude = x.max()
    random_am = torch.rand(mask.shape)*(max_amplitude*0.1)
    pertub_matrix = mask*random_am
    return x+pertub_matrix

def one_hot_encoding(X):
    X = [int(x) for x in X]
    n_values = 2
    b = np.eye(n_values)[X]
    return b

def DataTransform_TD(sample):
    """Augmentation bank that includes four augmentations and randomly select one as the positive sample.
    You may use this one the replace the above DataTransform_TD function."""
    aug_1 = add_noise(sample)
    aug_2 = time_mask(sample)
    li = np.random.randint(0, 2)
    aug_T = li * aug_1 + (1 - li) * aug_2
    return aug_T

def DataTransform_FD(sample):
    """Weak and strong augmentations in Frequency domain """
    aug_1 = remove_frequency(sample, pertub_ratio=0.1)
    aug_2 = add_frequency(sample, pertub_ratio=0.1)
    li = np.random.randint(0, 2)
    aug_F = li * aug_1 + (1 - li) * aug_2
    return aug_F



def bandpass_filter(signal, lowcut, highcut, fs, order=4):
    """
    带通滤波器
    参数:
    - signal: 输入信号
    - lowcut: 滤波器低截止频率
    - highcut: 滤波器高截止频率
    - fs: 信号采样率
    - order: 滤波器阶数
    返回:
    - filtered_signal: 滤波后的信号
    """
    nyquist = 0.5 * fs  # 奈奎斯特频率
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    filtered_signal = filtfilt(b, a, signal)
    return filtered_signal

def mark_rr_segments(rr_intervals, thre=0.6):
    """
    根据 RR 间期创建一个新的数组：
    - 其余位置是 0；
    - 连续小于 threshold 的值所在区段，每个位置替换为该区段的长度。

    参数:
        rr_intervals (numpy.ndarray): 输入的 RR 间期数组。
        threshold (int): 判断阈值，默认是 0.6s。

    返回:
        numpy.ndarray: 处理后的数组。
    """
    # 初始化输出数组，长度与 rr_intervals 相同，初始值为 0
    output_array = torch.zeros_like(rr_intervals)

    # 遍历 RR 间期数组，找出连续小于 threshold 的区段
    start_idx = None  # 当前连续区段的起始索引
    for i, value in enumerate(rr_intervals):
        if value < thre:
            if start_idx is None:
                start_idx = i  # 记录区段的起始索引
        else:
            if start_idx is not None:
                # 如果结束了一个连续小于 threshold 的区段，计算区段长度
                length = i - start_idx
                output_array[start_idx:i] = length  # 填充该区段长度
                start_idx = None  # 重置起始索引

    # 如果数组以一个连续区段结尾，需要处理最后一个区段
    if start_idx is not None:
        length = len(rr_intervals) - start_idx
        output_array[start_idx:] = length

    return output_array

def local_mean_fill(signal, flag, thre):
    """
    使用局部平均值填补信号中值为 0 的部分。
    
    参数:
    - signal: 输入信号 (numpy 数组)
    - flag: 标记每个位置连续<阈值的长度
    - thre: RR间期失效阈值
    
    返回:
    - filled_signal: 填补后的信号
    """
    filled_signal = signal.clone()  # 创建副本以进行填补

    # 遍历信号中的每个位置
    for i in range(len(signal)):
        if signal[i] < thre:  # 如果当前点为 0
            window_size = int(flag[i] * 1)
            # 确定局部窗口的范围
            start_idx = max(0, i - window_size)
            end_idx = min(len(signal), i + window_size + 1)

            # 提取窗口内的非零值
            local_values = signal[start_idx:end_idx]
            non_zero_values = local_values[local_values > thre]

            # 如果有非零值，使用其均值进行填补
            if len(non_zero_values) == 1:
                filled_signal[i] = non_zero_values[0]
            elif len(non_zero_values) > 1:
                filled_signal[i] = torch.mean(non_zero_values)

    return filled_signal


def detect_zero_segment(signal, threshold=0.0, min_len=10):
    """
    检测信号中是否存在连续的 0 段
    :param signal: 1D tensor
    :param threshold: 判断为0的阈值 (可以设为很小的数, 比如1e-5)
    :param min_len: 连续为0的最小长度
    :return: (缺失比例 zero_ratio, 位置掩码 valid_mask)
             valid_mask 是一个 bool tensor, 缺失段为 False, 其余为 True
    """
    signal = signal.clone()
    mask = (torch.abs(signal) <= threshold).int()
    diff = torch.diff(mask, prepend=torch.tensor([0], device=signal.device, dtype=torch.int))
    starts = torch.where(diff == 1)[0]
    ends = torch.where(diff == -1)[0]

    # 边界情况处理
    if mask[-1] == 1:
        ends = torch.cat([ends, torch.tensor([len(signal)], device=signal.device)])
    if len(starts) > 0 and (len(ends) == 0 or starts[0] > ends[0]):
        starts = torch.cat([torch.tensor([0], device=signal.device), starts])

    valid_mask = torch.ones_like(signal)
    total_len = 0
    for s, e in zip(starts, ends):
        if e - s >= min_len:
            valid_mask[s:e] = 0
            total_len += (e.item() - s.item())

    zero_ratio = total_len / len(signal)

    if isinstance(zero_ratio, torch.Tensor):
        aaa = 1

    return torch.tensor(zero_ratio, dtype=torch.float32).clone(), valid_mask

# 用于预训练的数据集构造
class ZHEdataset_Pretrain(Dataset):
    def __init__(self, data_dict, label_dict, seg_len, cut, flag, n_disea, fea_data, device):
        self.data_dict = data_dict
        self.label_dict = label_dict
        self.seg_len = seg_len
        self.cut = cut
        self.fea_data = fea_data
        self.fs_b = 10
        self.fs_p = 120
        self.fs_r = 0.5
        self.label_map = torch.tensor([[0,0],[1,0],[1,1],[0,1]], dtype=torch.float32)   # 第一位表示焦虑，第二位表示抑郁
        self.folds = StratifiedKFold(n_splits=4, shuffle=True, random_state=0)  # 4折交叉验证和类别数无关
        self.flag = flag
        self.n_disea = n_disea
        self.device = device

    def __len__(self):
        return len(self.data_dict)
 
    def __getitem__(self, index):
        data = self.data_dict[index]
        label = self.label_dict[index]
        x = data['Record_MicroHRRPdB1Hz']   # 1Hz
        p = data['ppg']    # 120Hz
        r = data['rri']    # 0.5Hz
        y = label['label']
        
        down_ratio = 4  # 降采样4倍到2.5Hz

        L = x.shape[0]

        rand_num = 0            
        while True:

            rand_num += 1
            seg_begin = random.randint(0, L - self.seg_len - 100)
            
            x = x[seg_begin:seg_begin+self.seg_len, :].reshape(-1)   # 10Hz
            r = r[:, seg_begin:seg_begin+self.seg_len]   # 1Hz  (3, n)
            p = p[seg_begin:seg_begin+self.seg_len, :].reshape(-1)   # 120Hz

            # 计算数据中失效的比例（PPG连续超过5min为0的片段）
            zr, mask_attn = detect_zero_segment(p.reshape(-1), 1e-3, 36000)

            if (zr < 0.1) or (rand_num > 10):
                break
        
        x_f = torch.fft.fft(x).abs()
        r_f = torch.fft.fft(r, dim=1).abs()
        p_f = torch.fft.fft(p).abs()

        x = x.reshape(1,-1)
        p = p.reshape(1,-1)
        x_f = x_f.reshape(1,-1)
        p_f = p_f.reshape(1,-1)
                    
        x = x[:, ::down_ratio]
        r = F.interpolate(r.unsqueeze(0), scale_factor=2.5, mode='linear').squeeze(0)
        x_f = x_f[:, ::down_ratio]
        r_f = F.interpolate(r_f.unsqueeze(0), scale_factor=2.5, mode='linear').squeeze(0)

        aug_x = DataTransform_TD(x)
        aug_r = DataTransform_TD(r)
        aug_p = DataTransform_TD(p)

        aug_x_f = DataTransform_FD(x_f)
        aug_r_f = DataTransform_FD(r_f)
        aug_p_f = DataTransform_FD(p_f)

    
        if x.shape[-1] * 48 != p.shape[-1]:
            a = 1
        if x.shape[-1] != r.shape[-1]:
            a = 1

        # print(f'types: x={type(x)}, r={type(r)}, p={type(p)}, f={type(f)}, s={type(s)}, y={type(y)}')
        
        # 不对样本加权的话 将zr都置为0
        zr = torch.tensor(0, dtype=torch.float32)

        return x, r, p, y, x_f, r_f, p_f, aug_x, aug_r, aug_p, aug_x_f, aug_r_f, aug_p_f, zr, mask_attn


class ZHEdataset(Dataset):
    def __init__(self, data_dict, seg_len, cut, flag, device, dynamic):

        self.data_dir = data_dict + 'processed_1028/'
        self.data_dict = os.listdir(data_dict + 'processed_1028/')  # 装有所有数据的文件夹
        self.map_label = {'非精神疾病': 0, '抑郁': 1, '焦虑': 2, '焦虑抑郁共病': 3}
        self.label_num = list(self.map_label.values())
        self.n_disea = len(self.map_label)
        self.fea_min = pickle.load(open(data_dict + 'fea_min.pkl', 'rb'))
        self.fea_max = pickle.load(open(data_dict + 'fea_max.pkl', 'rb'))
        self.dynamic = dynamic  # 该参数标记是否动态训练 也就是根据整晚数据长度分批训练
        self.fea_delete = [62, 63, 64, 65, 66, 67, 68, 69, 70, 78, 79, 80, 81, 82, 93, 94, 95, 96, 97, 108, 109, 110, 111, 112, 123, 124, 125, 126, 127]
        self.n_fea = 132 - len(self.fea_delete)   # 经验特征数目

        # Initialize lists to store valid data
        self.valid_files = []
        self.labels = []
        self.names = []
        self.lengths = []
        # Process each file in data_dict
        for filename in self.data_dict:
            # Split filename by '_' 
            parts = filename.split('_')
            if len(parts) >= 2:
                name = parts[0]
                label_str = parts[1]
                data_L = parts[4][:-4]
                # Check if label exists in mapping
                if label_str in self.map_label.keys():
                    # Add to valid data
                    self.valid_files.append(filename)
                    self.labels.append(self.map_label[label_str])
                    self.names.append(name)
                    self.lengths.append(int(data_L))
        # Convert lists to arrays
        self.labels = torch.tensor(self.labels)
        self.names = np.array(self.names)
        self.lengths = np.array(self.lengths)
        
        if self.dynamic:
        # Sort all lists based on lengths
            sorted_indices = np.argsort(self.lengths)
            self.valid_files = [self.valid_files[i] for i in sorted_indices]
            self.labels = self.labels[sorted_indices]
            self.names = self.names[sorted_indices]
            self.lengths = self.lengths[sorted_indices]

        self.norm_lengths = np.zeros(len(self.lengths))
        self.seg_len = seg_len
        self.cut = cut
        self.fs_b = 10
        self.fs_p = 120
        self.fs_r = 0.5
        self.fs_spec = 2.5
        self.label_map = torch.tensor([[1,0],[0,1],[1,1]], dtype=torch.float32)   # 第一位表示焦虑，第二位表示抑郁
        self.folds = StratifiedKFold(n_splits=4, shuffle=True, random_state=0)  # 4折交叉验证和类别数无关
        self.flag = flag
        self.device = device

 
    def __getitem__(self, index):
        data = pickle.load(open(self.data_dir + self.valid_files[index], 'rb'))
        y = self.labels[index]

        # 经验特征提取
        fea = np.array([float(x) for x in data['table_data']])
        f = torch.tensor(fea, dtype=torch.float32)
        if torch.sum(torch.isnan(f)).item() > 0:
            print(self.valid_files[index] + ' has nan')
        f = torch.nan_to_num(f, nan=0.0)
        f = (f - self.fea_min) / (self.fea_max - self.fea_min)
        equal_mask = (self.fea_max == self.fea_min)
        f[equal_mask] = 0.5

        # 根据 self.fea_delete 记录的下标去除 f 中指定的特征
        # 注意：self.fea_delete 是 list 类型，直接用列表索引
        if hasattr(self, 'fea_delete') and self.fea_delete is not None and len(self.fea_delete) > 0:
            f = torch.index_select(f, dim=0, index=torch.tensor([i for i in range(f.shape[0]) if i not in self.fea_delete], dtype=torch.long))

        x = data['breath']   # 10Hz shape:(N,)
        p = data['ppg']    # 120Hz
        r = data['rri_power']    # 1Hz
        s = data['sleep_stage_pkl']     # 1Hz
        
        down_ratio = 4  # 降采样4倍到2.5Hz
        up_ratio = 10  # 分期标签升采样10倍到10Hz
        
        if self.cut:

            if self.dynamic:
                seg_len = int(self.norm_lengths[index] * 4 / 4)  # 控制用于训练的数据长度的比例
                seg_len = max(seg_len - seg_len % 64, 64)  # 保证是64的倍数，且至少为64
                assert seg_len > 0, f'seg_len={seg_len} is not valid'

                # 获取最大裁剪起始点，防止超出原始长度
                L = r.shape[-1]  # 1Hz下的长度
                max_start = max(0, L - seg_len)
                seg_begin = random.randint(0, max_start) if max_start > 0 else 0

                x = x[int(self.fs_b*seg_begin):int(self.fs_b*seg_begin)+int(self.fs_b*seg_len)].reshape(-1)   # breath, 10Hz
                r = r[:, seg_begin:seg_begin+seg_len]   # rri_power, 1Hz  (3, n)
                p = p[seg_begin:seg_begin+seg_len, :].reshape(-1)   # ppg, 120Hz
                s = s[seg_begin:seg_begin+seg_len]    # sleep_stage_pkl, 1Hz

            else:
                L = r.shape[-1]  # 1Hz长度
                seg_len = self.seg_len
                seg_begin = random.randint(0, L - seg_len - 100)

                x = x[int(self.fs_b*seg_begin):int(self.fs_b*seg_begin)+int(self.fs_b*seg_len)].reshape(-1)   # 10Hz
                r = r[:, seg_begin:seg_begin+seg_len]   # 1Hz  (3, n)
                p = p[seg_begin:seg_begin+seg_len, :].reshape(-1)   # 120Hz
                s = s[seg_begin:seg_begin+seg_len]    # 1Hz

            # print(seg_len)
            # 计算数据中失效的比例（PPG连续超过5min为0的片段）
            zr, mask_attn = detect_zero_segment(p.reshape(-1), 1e-3, 36000)

            if torch.rand(1).item() < 0.8:  # 1/2的概率随机缩放   不使用这个数据增强的话就设成0
                scale_factor = 0.5 * torch.rand(1).item() + 0.75
                xx = time_scaling(x, scale_factor, 'linear')
                rr = time_scaling(r, scale_factor, 'linear')  # (3, n)
                pp = time_scaling(p, scale_factor, 'linear')
                ss = time_scaling(s, scale_factor, 'nearest')
                mm = time_scaling(mask_attn.float(), scale_factor, 'nearest')
                if scale_factor >= 1:
                    x = xx[:int(seg_len*self.fs_b)].reshape(1,-1)
                    r = rr[:, :seg_len]
                    p = pp[:seg_len*self.fs_p].reshape(1,-1)
                    mask_attn = mm[:seg_len*self.fs_p]
                    s = ss[:seg_len]
                else:
                    delta_x = x.shape[0] - xx.shape[0]
                    delta_r = r.shape[-1] - rr.shape[-1]
                    delta_p = p.shape[0] - pp.shape[0]
                    delta_m = mask_attn.shape[0] - mm.shape[0]
                    delta_s = s.shape[0] - ss.shape[0]
                    x = torch.cat([xx, xx[-delta_x:]]).reshape(1,-1)
                    r = torch.cat([rr, rr[:, -delta_r:]], dim=1)  # (3, n)
                    p = torch.cat([pp, pp[-delta_p:]]).reshape(1,-1)
                    mask_attn = torch.cat([mm, mm[-delta_m:]])
                    s = torch.cat([ss, ss[-delta_s:]])
                    if x.shape[-1] != int(seg_len*self.fs_b):
                        aaa = 1
                    assert x.shape[-1] == int(seg_len*self.fs_b)
                    assert r.shape[-1] == seg_len
                    assert p.shape[-1] == seg_len*self.fs_p
                    assert mask_attn.shape[0] == seg_len*self.fs_p
                    assert s.shape[0] == seg_len
            else:
                x = x.reshape(1,-1)
                p = p.reshape(1,-1)

            if torch.rand(1).item() > 0.5:
                x = add_noise(x)
                r = add_noise(r)
                p = add_noise(p)
            
            if torch.rand(1).item() < 0.5:
                x = time_mask(x)
                r = time_mask(r)
                p = time_mask(p)

            s = s.repeat_interleave(up_ratio)
            r = F.interpolate(r.unsqueeze(0), scale_factor=2.5, mode='linear').squeeze(0)
            s = s[::down_ratio]

        else:  # 验证集使用整晚数据时，需要补零
            # 计算数据中失效的比例（PPG连续超过5min为0的片段）
            zr, mask_attn = detect_zero_segment(p.reshape(-1), 1e-3, 36000)
            x = x.reshape(1, -1)  # 10Hz
            r = F.interpolate(r.unsqueeze(0), scale_factor=2.5, mode='linear').squeeze(0)
            p = p.reshape(1, -1)
            s = s.repeat_interleave(up_ratio)

            x = pad_to_multiple(x, 128)
            r = pad_to_multiple(r, 32)
            s = pad_to_multiple(s[::down_ratio], 32)
            p = pad_to_multiple(p, 1536)
            mask_attn = pad_to_multiple(mask_attn, 1536)
        
        if x.shape[-1] * 12 != p.shape[-1]:
            aaa = 1
        assert x.shape[-1] * 12 == p.shape[-1]
        assert x.shape[-1] * 12 == mask_attn.shape[-1]
        assert x.shape[-1] == r.shape[-1] * 4

        # 多标签损失映射
        if self.flag:
            y = self.label_map[int(y.item())]

        # print(f'types: x={type(x)}, r={type(r)}, p={type(p)}, f={type(f)}, s={type(s)}, y={type(y)}')
        
        # 不对样本加权的话 将zr都置为0
        zr = torch.tensor(0, dtype=torch.float32)
        # y_qst = torch.zeros(16, dtype=torch.float32)
        # mask_qst = torch.zeros(16, dtype=torch.float32)


        return x, r, p, f, s, y, zr, mask_attn

    def split(self, fold, batch_size):
        # fold 代表所用的折数  用于交叉验证划分train_index, valid_index
        # 构建每个人的标签字典
        person_labels = {}
        for i in range(len(self.labels)):
            name = self.names[i]
            label = self.labels[i]
            if name not in person_labels:
                person_labels[name] = {'indices': [], 'label': label}
            person_labels[name]['indices'].append(i)

        names = list(person_labels.keys())
        labels = [person_labels[name]['label'] for name in names]
        
        # 存储每折对应的样本索引
        fold_indices = []
        for train_names_idx, val_names_idx in self.folds.split(names, labels):
            # 获取当前折的训练集和验证集人名
            train_names = [names[i] for i in train_names_idx]
            val_names = [names[i] for i in val_names_idx]
            
            # 将人名映射回原始数据的索引
            train_indices = []
            val_indices = []
            for name in train_names:
                train_indices.extend(person_labels[name]['indices'])
            for name in val_names:
                val_indices.extend(person_labels[name]['indices'])
                
            fold_indices.append((sorted(train_indices), sorted(val_indices)))
        
        counter = Counter(list(np.array(self.labels)))
        num_list = [counter[i] for i in range(len(counter))]
        
        s_num_list = [18000, 37000, 9000, 14000, 20000, 4000]  # 经验定义的分期数目列表

        train_index, val_index = fold_indices[fold]
        length_array = np.array(self.lengths)
        train_length_array = length_array[np.array(train_index)]
        val_length_array = length_array[np.array(val_index)]

        for i in range(math.ceil(len(train_length_array) / batch_size)):
            left_index = i * batch_size
            right_index = min(left_index + batch_size, len(train_length_array))
            # 计算norm_length，使其小于等于min(train_length_array[left_index:right_index])
            # 并且2.5*norm_length是32的倍数，120*norm_length是1536的倍数
            min_len = min(train_length_array[left_index:right_index])
            # 2.5*norm_length = 32*k1 => norm_length = 32*k1/2.5 = 12.8*k1
            # 120*norm_length = 1536*k2 => norm_length = 12.8*k2
            # 所以norm_length必须是12.8的倍数，且<=min_len  12.8*5=64
            # 取最大的k使得norm_length<=min_len
            max_k = int(min_len // 64)
            norm_length = int(64 * max_k)
            self.norm_lengths[train_index[left_index:right_index]] = norm_length

        return train_index, val_index, num_list, s_num_list
    
    # def split(self, fold, batch_size):
    #     # fold 代表所用的折数  用于交叉验证划分train_index, valid_index
    #     # 构建每个人的标签字典和索引列表
    #     person_labels = {}
    #     for i in range(len(self.labels)):
    #         name = self.names[i]
    #         label = self.labels[i]
    #         if name not in person_labels:
    #             person_labels[name] = {'indices': [], 'label': label}
    #         person_labels[name]['indices'].append(i)

    #     # 对每个人的数据进行随机划分
    #     train_indices = []
    #     val_indices = []
        
    #     for name in person_labels:
    #         indices = person_labels[name]['indices']
    #         # 随机打乱该人的所有样本索引
    #         random.shuffle(indices)
    #         # 计算验证集大小(取20%作为验证集)
    #         val_size = int(len(indices) * 0.5)
    #         # 划分训练集和验证集
    #         person_val_indices = indices[:val_size]
    #         person_train_indices = indices[val_size:]
            
    #         train_indices.extend(person_train_indices)
    #         val_indices.extend(person_val_indices)

    #     # 统计原始标签分布
    #     counter = Counter(list(np.array(self.labels)))
    #     num_list = [counter[i] for i in range(len(counter))]
        
    #     # 经验定义的分期数目列表
    #     s_num_list = [18000, 37000, 9000, 14000, 20000, 4000]

    #     train_index, val_index = sorted(train_indices), sorted(val_indices)
    #     length_array = np.array(self.lengths)
    #     train_length_array = length_array[np.array(train_index)]
    #     val_length_array = length_array[np.array(val_index)]

    #     for i in range(math.ceil(len(train_length_array) / batch_size)):
    #         left_index = i * batch_size
    #         right_index = min(left_index + batch_size, len(train_length_array))
    #         # 计算norm_length，使其小于等于min(train_length_array[left_index:right_index])
    #         # 并且2.5*norm_length是32的倍数，120*norm_length是1536的倍数
    #         min_len = min(train_length_array[left_index:right_index])
    #         # 2.5*norm_length = 32*k1 => norm_length = 32*k1/2.5 = 12.8*k1
    #         # 120*norm_length = 1536*k2 => norm_length = 12.8*k2
    #         # 所以norm_length必须是12.8的倍数，且<=min_len  12.8*5=64
    #         # 取最大的k使得norm_length<=min_len
    #         max_k = int(min_len // 64)
    #         norm_length = int(64 * max_k)
    #         self.norm_lengths[train_index[left_index:right_index]] = norm_length

    #     return train_index, val_index, num_list, s_num_list