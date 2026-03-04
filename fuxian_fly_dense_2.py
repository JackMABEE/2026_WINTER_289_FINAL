import numpy as np
import matplotlib.pyplot as plt
import os
import requests
import h5py
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors

# ==========================================
# 1. 核心哈希算法
# ==========================================
def lsh_hash(X, m):
    """标准的 Dense LSH：使用高斯随机超平面，50% 激活"""
    np.random.seed(42)
    W = np.random.randn(X.shape[1], m)
    return (X @ W > 0).astype(int)

def fly_hash_biological(X, m, k, synapse_sparsity=0.1):
    """
    完美的生物学 FlyHash (复现 Dasgupta 2017):
    1. 二值稀疏投影 (Binary Sparse Projection)
    2. WTA 赢者通吃 (Top-k)
    """
    np.random.seed(42)
    # 【核心改动】：生成 0 和 1 的二值矩阵，代表生物突触的连接与否！
    # 假设每个神经元只与约 10% 的输入神经元相连
    W_binary = (np.random.rand(X.shape[1], m) < synapse_sparsity).astype(float)
    
    # 突触信号传递（仅仅是加和，没有负权重的干扰）
    activations = X @ W_binary
    
    # WTA 机制：只保留最兴奋的 Top-k 神经元
    binary_hashes = np.zeros_like(activations, dtype=int)
    top_k_indices = np.argsort(activations, axis=1)[:, -k:]
    rows = np.arange(X.shape[0])[:, None]
    binary_hashes[rows, top_k_indices] = 1
    return binary_hashes

def compute_map(true_nn, query_hashes, db_hashes):
    # 对于固定个数 k 个 1 的二值向量，点积就是“重合神经元的个数”（交集大小），
    # 这完全契合 Dasgupta 原论文的距离度量。
    similarities = query_hashes @ db_hashes.T 
    sorted_retrieval = np.argsort(-similarities, axis=1)
    aps = []
    for i in range(true_nn.shape[0]):
        truth_set = set(true_nn[i])
        retrieved_list = sorted_retrieval[i]
        hits = 0
        sum_precisions = 0.0
        for rank, doc_id in enumerate(retrieved_list[:100]):
            if doc_id in truth_set:
                hits += 1
                sum_precisions += hits / (rank + 1.0)
        aps.append(sum_precisions / len(truth_set) if len(truth_set) > 0 else 0)
    return np.mean(aps)

# ==========================================
# 2. 稳健的数据下载与加载
# ==========================================
def download_file(url, filename):
    if os.path.exists(filename):
        return True
    print(f"  [!] 正在下载 {filename} ...")
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        with requests.get(url, headers=headers, stream=True) as r:
            r.raise_for_status()
            with open(filename, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        return True
    except Exception as e:
        if os.path.exists(filename): os.remove(filename)
        return False

def get_data(name):
    print(f"\n>>> 准备 {name} 数据集 <<<")
    if name == 'MNIST':
        mnist = fetch_openml('mnist_784', version=1, as_frame=False)
        X = mnist.data[:10000]
    else:
        urls = {
            'SIFT': 'http://ann-benchmarks.com/sift-128-euclidean.hdf5',
            'GloVe': 'http://ann-benchmarks.com/glove-100-angular.hdf5'
        }
        filename = f"{name.lower()}_data.hdf5"
        if not download_file(urls[name], filename):
            raise RuntimeError(f"无法获取 {name} 数据集。")
        with h5py.File(filename, 'r') as f:
            X = np.array(f['train'][:10000])
            
    # 【核心】：仅 Mean-centering，还原真实物理流形
    X = X - np.mean(X, axis=0)
    
    X_train, X_test = train_test_split(X, test_size=100, random_state=42)
    nbrs = NearestNeighbors(n_neighbors=10).fit(X_train)
    _, gt = nbrs.kneighbors(X_test)
    return X_train, X_test, gt

# ==========================================
# 3. 运行复现实验
# ==========================================
datasets = ['MNIST', 'SIFT', 'GloVe']
# 【核心改动】：加入了 10000 维的极端膨胀！
m_values = [1000, 2000, 4000, 10000]
results = {ds: {'fly': [], 'dense': []} for ds in datasets}

K_RATIO = 0.10 # 10% 稀疏度

for ds in datasets:
    xt, xq, gt = get_data(ds)
    for m in m_values:
        print(f"    --> 测试 m={m} ...")
        k_val = max(1, int(m * K_RATIO))
        # Dense LSH
        results[ds]['dense'].append(compute_map(gt, lsh_hash(xq, m), lsh_hash(xt, m)))
        # Biological FlyHash
        results[ds]['fly'].append(compute_map(gt, fly_hash_biological(xq, m, k_val), fly_hash_biological(xt, m, k_val)))

# ==========================================
# 4. 绘图
# ==========================================
print("\n>>> 实验完成，正在生成图表... <<<")
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for i, ds in enumerate(datasets):
    axes[i].plot(m_values, results[ds]['dense'], marker='o', linestyle='-', linewidth=2, label='Dense LSH (50%)', color='#3498db')
    axes[i].plot(m_values, results[ds]['fly'], marker='s', linestyle='-', linewidth=2, label='Biological FlyHash (10%)', color='#e67e22')
    
    axes[i].set_title(f'{ds} Dataset\n(No L2 Norm, Binary Synapses)', fontsize=14, fontweight='bold')
    axes[i].set_xlabel('Hash Length ($m$)', fontsize=12)
    if i == 0:
        axes[i].set_ylabel('Mean Average Precision (mAP)', fontsize=12)
    axes[i].set_xticks(m_values)
    axes[i].legend(fontsize=11)
    axes[i].grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.savefig('perfect_science_reproduction.png', dpi=300, bbox_inches='tight')
print("\n[SUCCESS] 图表已保存！")
plt.show()