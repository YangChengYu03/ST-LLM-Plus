"""
数据集模块
- StandardScaler: Z-Score 标准化器（支持按节点独立标准化）
- STTrafficDataset: 时空交通数据集（支持可配置 stride）
- create_dataloaders: 创建训练/验证/测试 DataLoader
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class StandardScaler:
    """
    Z-Score 标准化器

    支持两种模式：
    - 全局标准化 (per_node=False): 对所有时间步和节点计算统一均值/标准差
    - 按节点标准化 (per_node=True): 对每个节点独立计算均值/标准差，保留空间差异性
    """

    def __init__(self, per_node=False):
        self.per_node = per_node
        self.mean = 0.0
        self.std = 1.0

    def fit(self, data):
        """
        在训练集上拟合

        Args:
            data: numpy array, shape (T, N, C)
        """
        if self.per_node:
            # 按节点独立标准化: axis=0 (时间维度)
            self.mean = np.mean(data, axis=0, keepdims=True)  # (1, N, C)
            self.std = np.std(data, axis=0, keepdims=True)    # (1, N, C)
        else:
            # 全局标准化
            self.mean = np.mean(data, axis=(0, 1), keepdims=True)  # (1, 1, C)
            self.std = np.std(data, axis=(0, 1), keepdims=True)    # (1, 1, C)

        # 防止除以零
        self.std[self.std == 0] = 1.0

    def transform(self, data):
        """标准化"""
        return (data - self.mean) / self.std

    def inverse_transform(self, data, feature_idx=0):
        """
        反标准化（将预测值还原为真实量级）

        Args:
            data: numpy array 或 torch.Tensor
            feature_idx: 要反归一化的特征索引

        Returns:
            反标准化后的数据
        """
        if isinstance(data, torch.Tensor):
            mean_val = torch.tensor(self.mean.flat[feature_idx], dtype=data.dtype, device=data.device)
            std_val = torch.tensor(self.std.flat[feature_idx], dtype=data.dtype, device=data.device)
        else:
            if self.per_node:
                # per_node 模式下 mean/std 形状为 (1, N, C)
                mean_val = self.mean[0, :, feature_idx]  # (N,)
                std_val = self.std[0, :, feature_idx]     # (N,)
            else:
                mean_val = self.mean[0, 0, feature_idx]
                std_val = self.std[0, 0, feature_idx]

        return data * std_val + mean_val


class STTrafficDataset(Dataset):
    """
    时空交通数据集

    滑窗式生成 (X, Y) 样本对：
    - X: (T_in, N, C) 历史时间步的全部特征
    - Y: (T_out, N, 1) 未来时间步的目标特征（拥堵指数）

    Args:
        data: numpy array, shape (T, N, C)
        T_in: 历史输入步长
        T_out: 预测输出步长
        stride: 滑窗步长
    """

    def __init__(self, data, T_in=12, T_out=12, stride=1):
        self.data = data
        self.T_in = T_in
        self.T_out = T_out
        self.stride = stride

        total_window = T_in + T_out
        self.num_samples = max(0, (data.shape[0] - total_window) // stride + 1)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        start = index * self.stride
        X = self.data[start: start + self.T_in, :, :]
        Y = self.data[start + self.T_in: start + self.T_in + self.T_out, :, 0:1]
        return torch.FloatTensor(X), torch.FloatTensor(Y)


def create_dataloaders(args, scaler_per_node=False):
    """
    创建训练/验证/测试 DataLoader

    Args:
        args: 配置参数
        scaler_per_node: 是否按节点独立标准化

    Returns:
        train_loader, val_loader, test_loader, scaler
    """
    print(f"正在加载时空张量数据: {args.npz_path}")
    data = np.load(args.npz_path)['data']  # (T, N, C)
    T_total, N, C = data.shape
    print(f"数据总览 -> 时间步:{T_total}, 节点数:{N}, 特征维:{C}")

    # 数据划分
    train_steps = int(T_total * args.train_ratio)
    val_steps = int(T_total * args.val_ratio)

    T_in = args.input_len
    train_data = data[:train_steps]
    val_data = data[train_steps - T_in: train_steps + val_steps]
    test_data = data[train_steps + val_steps - T_in:]

    # Z-Score 标准化
    scaler = StandardScaler(per_node=scaler_per_node)
    scaler.fit(train_data)

    train_data = scaler.transform(train_data)
    val_data = scaler.transform(val_data)
    test_data = scaler.transform(test_data)

    # 创建 Dataset
    train_dataset = STTrafficDataset(train_data, T_in, args.output_len, stride=args.stride)
    val_dataset = STTrafficDataset(val_data, T_in, args.output_len, stride=1)
    test_dataset = STTrafficDataset(test_data, T_in, args.output_len, stride=1)

    # 创建 DataLoader
    num_workers = getattr(args, "num_workers", 0)
    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, drop_last=True, num_workers=num_workers,
                              pin_memory=pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=num_workers,
                            pin_memory=pin_memory)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                             shuffle=False, num_workers=num_workers,
                             pin_memory=pin_memory)

    print(f"✅ DataLoader 创建完成！")
    print(f"   训练集: {len(train_dataset)} 样本")
    print(f"   验证集: {len(val_dataset)} 样本")
    print(f"   测试集: {len(test_dataset)} 样本")

    return train_loader, val_loader, test_loader, scaler
