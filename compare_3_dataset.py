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
    np.random.seed(42)
    W = np.random.randn(X.shape[1], m)
    return (X @ W > 0).astype(int)

def fly_hash(X, m, k, sparsity=0.1):
    np.random.seed(42)
    W = np.random.randn(X.shape[1], m)
    mask = (np.random.rand(X.shape[1], m) < sparsity).astype(float)
    activations = X @ (W * mask)
    binary_hashes = np.zeros_like(activations, dtype=int)
    top_k_indices = np.argsort(activations, axis=1)[:, -k:]
    rows = np.arange(X.shape[0])[:, None]
    binary_hashes[rows, top_k_indices] = 1
    return binary_hashes

def threshold_hash(X, m, theta, sparsity=0.1):
    np.random.seed(42)
    W = np.random.randn(X.shape[1], m)
    mask = (np.random.rand(X.shape[1], m) < sparsity).astype(float)
    activations = X @ (W * mask)
    return (activations > theta).astype(int)

def compute_map(true_nn, query_hashes, db_hashes):
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
    """使用 requests 进行伪装和流式下载，防 403 报错"""
    if os.path.exists(filename):
        print(f"  [+] {filename} 已存在，跳过下载。")
        return True
    
    print(f"  [!] 正在从 ANN-Benchmarks 下载 {filename} ...")
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0.0.0 Safari/537.36'}
    try:
        with requests.get(url, headers=headers, stream=True) as r:
            r.raise_for_status()
            total_size = int(r.headers.get('content-length', 0))
            downloaded_size = 0
            with open(filename, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
                    downloaded_size += len(chunk)
                    # 简易进度条
                    if total_size > 0 and downloaded_size % (1024 * 1024 * 10) < 8192: # 每 10MB 打印一次
                        print(f"      已下载: {downloaded_size / (1024*1024):.1f} MB / {total_size / (1024*1024):.1f} MB")
        print(f"  [+] {filename} 下载完成！")
        return True
    except Exception as e:
        print(f"  [-] 下载失败: {e}")
        if os.path.exists(filename):
            os.remove(filename) # 删除下载残缺的文件
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
            raise RuntimeError(f"无法获取 {name} 数据集，请检查网络。")
            
        print(f"  [+] 正在读取 HDF5 文件...")
        with h5py.File(filename, 'r') as f:
            X = np.array(f['train'][:10000]) # 提取前 10000 条作为数据库
            
    print(f"  [+] 数据预处理 (Mean-centering & L2 Normalization)...")
    X = X - np.mean(X, axis=0)
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
    X_train, X_test = train_test_split(X, test_size=100, random_state=42)
    
    print(f"  [+] 计算 Ground Truth (Exact KNN)...")
    nbrs = NearestNeighbors(n_neighbors=10).fit(X_train)
    _, gt = nbrs.kneighbors(X_test)
    return X_train, X_test, gt

# ==========================================
# 3. 运行多数据集实验
# ==========================================
datasets = ['MNIST', 'SIFT', 'GloVe']
m_values = [500, 1000, 2000, 4000]
results = {ds: {'fly': [], 'thresh': [], 'dense': []} for ds in datasets}

# 统一参数：目标稀疏度均为 10%
K_RATIO = 0.10
THETA = 0.10  

for ds in datasets:
    xt, xq, gt = get_data(ds)
    for m in m_values:
        print(f"    --> 测试 m={m} ...")
        k_val = max(1, int(m * K_RATIO))
        # Dense LSH (50% bits)
        results[ds]['dense'].append(compute_map(gt, lsh_hash(xq, m), lsh_hash(xt, m)))
        # FlyHash (10% bits via WTA)
        results[ds]['fly'].append(compute_map(gt, fly_hash(xq, m, k_val), fly_hash(xt, m, k_val)))
        # ThresholdHash (approx 10% bits via Manifold adaptivity)
        results[ds]['thresh'].append(compute_map(gt, threshold_hash(xq, m, THETA), threshold_hash(xt, m, THETA)))

# ==========================================
# 4. 绘制终极对比大图
# ==========================================
print("\n>>> 实验完成，正在生成图表... <<<")
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for i, ds in enumerate(datasets):
    axes[i].plot(m_values, results[ds]['dense'], marker='o', linestyle='-', linewidth=2, label='Dense LSH (50% bits)', color='#3498db')
    axes[i].plot(m_values, results[ds]['fly'], marker='s', linestyle='-', linewidth=2, label='FlyHash (10% target)', color='#e67e22')
    axes[i].plot(m_values, results[ds]['thresh'], marker='^', linestyle='-', linewidth=2, label=f'ThresholdHash ($\\theta$={THETA})', color='#2ecc71')
    
    axes[i].set_title(f'{ds} Dataset', fontsize=14, fontweight='bold')
    axes[i].set_xlabel('Hash Length ($m$)', fontsize=12)
    if i == 0:
        axes[i].set_ylabel('Mean Average Precision (mAP)', fontsize=12)
    axes[i].set_xticks(m_values)
    axes[i].legend(fontsize=10)
    axes[i].grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.savefig('cross_dataset_evaluation.png', dpi=300, bbox_inches='tight')
print("\n[SUCCESS] 图表已保存为 'cross_dataset_evaluation.png'！")
plt.show()