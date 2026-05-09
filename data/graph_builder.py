"""
图结构构建模块
- 物理距离图：基于 Haversine 距离 + 高斯核
- 语义相似图：基于 POI/NTL 特征的余弦相似度
"""

import numpy as np
import pandas as pd
import os
from sklearn.metrics.pairwise import cosine_similarity


def _haversine_vectorized(lat1, lon1, lat2, lon2):
    """
    向量化的 Haversine 距离计算
    输入：弧度制的经纬度数组
    输出：距离矩阵 (km)
    """
    R = 6371.0  # 地球半径 (公里)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c


def build_dist_graph(npz_path, coords_csv_path, save_path, sigma=5.0, epsilon=0.1):
    """
    构建物理距离邻接矩阵

    Args:
        npz_path: 数据 npz 文件路径（用于获取节点数和节点排序）
        coords_csv_path: 路段坐标 CSV 文件路径
        save_path: 保存路径
        sigma: 高斯核 sigma
        epsilon: 稀疏化阈值

    Returns:
        A_dist: (N, N) 距离邻接矩阵
    """
    print(f"正在读取 {npz_path} 以获取节点维度...")
    data = np.load(npz_path)['data']
    T, N, C = data.shape

    print(f"正在处理轨迹文件 {coords_csv_path} 计算路段中心点...")
    df_coords = pd.read_csv(coords_csv_path)

    # 计算每一小段线段的中点
    df_coords['mid_x'] = (df_coords['START_X'] + df_coords['END_X']) / 2.0
    df_coords['mid_y'] = (df_coords['START_Y'] + df_coords['END_Y']) / 2.0

    # 按路段分组取平均得到几何中心
    centroids = df_coords.groupby('roadsegid')[['mid_y', 'mid_x']].mean().reset_index()
    centroids.rename(columns={'mid_y': 'lat', 'mid_x': 'lon'}, inplace=True)

    # 对齐节点顺序
    sorted_nodes = sorted(centroids['roadsegid'].unique())
    if len(sorted_nodes) != N:
        print(f"⚠️ 警告：npz节点数 ({N}) 与坐标CSV路段数 ({len(sorted_nodes)}) 不一致！")

    centroids = centroids.set_index('roadsegid').reindex(sorted_nodes)

    # 处理缺失坐标
    missing_mask = centroids['lat'].isnull().values
    if missing_mask.any():
        print(f"⚠️ 警告：有 {missing_mask.sum()} 个路段缺少坐标，将进行惩罚处理。")
        centroids = centroids.fillna(0)

    lat_lon = centroids[['lat', 'lon']].values

    # =====================================================
    # 向量化计算 Haversine 距离矩阵（取代双重 for 循环）
    # =====================================================
    print("开始计算路段中心点之间的物理距离矩阵 (向量化)...")
    lat_rad = np.radians(lat_lon[:, 0])
    lon_rad = np.radians(lat_lon[:, 1])

    # 构建 N×N 的差异矩阵
    lat1 = lat_rad[:, np.newaxis]  # (N, 1)
    lat2 = lat_rad[np.newaxis, :]  # (1, N)
    lon1 = lon_rad[:, np.newaxis]  # (N, 1)
    lon2 = lon_rad[np.newaxis, :]  # (1, N)

    dist_matrix = _haversine_vectorized(lat1, lon1, lat2, lon2).astype(np.float32)

    # 处理缺失坐标的惩罚
    for i in range(N):
        if missing_mask[i]:
            dist_matrix[i, :] = 99999.0
            dist_matrix[:, i] = 99999.0

    # 高斯核函数转换与稀疏化
    print("应用高斯核函数...")
    A_dist = np.exp(-(dist_matrix ** 2) / (sigma ** 2))
    A_dist[A_dist < epsilon] = 0.0
    np.fill_diagonal(A_dist, 1.0)

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    np.save(save_path, A_dist)
    print(f"✅ 物理距离图 A_dist 构建完成！形状: {A_dist.shape}，已保存至 {save_path}")

    return A_dist


def build_semantic_graph(npz_path, save_path, poi_features=None, ntl_feature=7,
                         threshold=0.5):
    """
    构建语义相似邻接矩阵

    Args:
        npz_path: 数据 npz 文件路径
        save_path: 保存路径
        poi_features: POI 特征索引列表，如 [4, 5]
        ntl_feature: NTL 特征索引
        threshold: 相似度阈值

    Returns:
        A_sem: (N, N) 语义邻接矩阵
    """
    if poi_features is None:
        poi_features = [4, 5]

    print(f"正在构建语义相似图...")
    data = np.load(npz_path)['data']

    # 提取特征：POI + NTL
    feature_indices = poi_features + [ntl_feature]
    node_features = data[0, :, feature_indices]  # (N, F)

    # 确保形状正确
    if node_features.shape[0] == len(feature_indices) and node_features.shape[1] != len(feature_indices):
        node_features = node_features.T

    print(f"  特征矩阵形状: {node_features.shape}")

    # 计算余弦相似度
    A_sem = cosine_similarity(node_features)
    A_sem = np.clip(A_sem, 0, 1).astype(np.float32)
    A_sem[A_sem < threshold] = 0.0
    np.fill_diagonal(A_sem, 1.0)

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    np.save(save_path, A_sem)
    print(f"✅ 语义相似图 A_sem 构建完成！形状: {A_sem.shape}，已保存至 {save_path}")

    return A_sem
