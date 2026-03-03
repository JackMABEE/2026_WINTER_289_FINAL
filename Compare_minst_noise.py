import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.neighbors import NearestNeighbors
import time

# ==========================================
# 1. Core Hashing Functions
# ==========================================
def lsh_hash(X, m):
    np.random.seed(42)
    W = np.random.randn(X.shape[1], m)
    return (X @ W > 0).astype(int)

def fly_hash(X, m, k, sparsity=0.1):
    np.random.seed(42)
    W = np.random.randn(X.shape[1], m)
    mask = (np.random.rand(X.shape[1], m) < sparsity).astype(float)
    W = W * mask
    activations = X @ W
    binary_hashes = np.zeros_like(activations, dtype=int)
    top_k_indices = np.argsort(activations, axis=1)[:, -k:]
    rows = np.arange(X.shape[0])[:, None]
    binary_hashes[rows, top_k_indices] = 1
    return binary_hashes

def threshold_hash(X, m, threshold, sparsity=0.1):
    np.random.seed(42)
    W = np.random.randn(X.shape[1], m)
    mask = (np.random.rand(X.shape[1], m) < sparsity).astype(float)
    W = W * mask
    activations = X @ W
    return (activations > threshold).astype(int)

def compute_map(true_nn, query_hashes, db_hashes):
    similarities = query_hashes @ db_hashes.T 
    sorted_retrieval = np.argsort(-similarities, axis=1)
    
    aps = []
    for i in range(true_nn.shape[0]):
        truth_set = set(true_nn[i])
        retrieved_list = sorted_retrieval[i]
        hits = 0
        sum_precisions = 0.0
        for rank, doc_id in enumerate(retrieved_list[:1000]):
            if doc_id in truth_set:
                hits += 1
                sum_precisions += hits / (rank + 1.0)
                if hits == len(truth_set):
                    break
        aps.append(sum_precisions / len(truth_set) if hits > 0 else 0.0)
    return np.mean(aps)

# ==========================================
# 2. Data Preparation Helper
# ==========================================
def prepare_and_get_ground_truth(X_raw, db_size=10000, query_size=100, top_n=50):
    X_db = X_raw[:db_size]
    X_query = X_raw[db_size:db_size + query_size]
    
    # Mean-centering and L2 Normalization (Crucial for fair comparison)
    mean_vec = np.mean(X_db, axis=0)
    X_db = X_db - mean_vec
    X_query = X_query - mean_vec
    X_db = X_db / (np.linalg.norm(X_db, axis=1, keepdims=True) + 1e-9)
    X_query = X_query / (np.linalg.norm(X_query, axis=1, keepdims=True) + 1e-9)
    
    # Ground Truth Exact Nearest Neighbors
    nbrs = NearestNeighbors(n_neighbors=top_n, algorithm='brute', metric='euclidean')
    nbrs.fit(X_db)
    _, gt_indices = nbrs.kneighbors(X_query)
    
    return X_db, X_query, gt_indices

# ==========================================
# 3. Evaluation Pipeline
# ==========================================
def evaluate_dataset(X_db, X_query, gt_indices, m_values, k_ratio, theta):
    lsh_maps, fly_maps, thresh_maps = [], [], []
    for m in m_values:
        print(f"    Evaluating m = {m}...")
        
        # Dense LSH
        db_lsh = lsh_hash(X_db, m)
        q_lsh = lsh_hash(X_query, m)
        lsh_maps.append(compute_map(gt_indices, q_lsh, db_lsh))
        
        # FlyHash
        optimal_k = max(1, int(m * k_ratio))
        db_fly = fly_hash(X_db, m, optimal_k)
        q_fly = fly_hash(X_query, m, optimal_k)
        fly_maps.append(compute_map(gt_indices, q_fly, db_fly))
        
        # ThresholdHash
        db_thresh = threshold_hash(X_db, m, theta)
        q_thresh = threshold_hash(X_query, m, theta)
        thresh_maps.append(compute_map(gt_indices, q_thresh, db_thresh))
        
    return lsh_maps, fly_maps, thresh_maps

# ==========================================
# 4. Main Experiment Execution
# ==========================================
OPTIMAL_K_RATIO = 0.1
OPTIMAL_THETA = 0.1
m_values = [500, 1000, 2000, 4000, 10000]

print(">>> Generating Synthetic Uniform Random Noise Dataset...")
# 生成 10100 个 784 维的纯随机向量（没有任何聚类和流形结构）
X_random_raw = np.random.rand(10100, 784)
X_db_rand, X_q_rand, gt_rand = prepare_and_get_ground_truth(X_random_raw)

print(">>> Downloading Real World Manifold Dataset (MNIST)...")
mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='liac-arff')
X_mnist_raw = mnist.data
X_db_mnist, X_q_mnist, gt_mnist = prepare_and_get_ground_truth(X_mnist_raw)

print("\n--- Running Evaluation on MNIST (Manifold Present) ---")
lsh_mnist, fly_mnist, thresh_mnist = evaluate_dataset(X_db_mnist, X_q_mnist, gt_mnist, m_values, OPTIMAL_K_RATIO, OPTIMAL_THETA)

print("\n--- Running Evaluation on Random Noise (No Manifold) ---")
lsh_rand, fly_rand, thresh_rand = evaluate_dataset(X_db_rand, X_q_rand, gt_rand, m_values, OPTIMAL_K_RATIO, OPTIMAL_THETA)

# ==========================================
# 5. Plotting Side-by-Side Comparison
# ==========================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Plot 1: MNIST (Manifold Data)
ax1.plot(m_values, lsh_mnist, marker='o', label='Dense LSH', linewidth=2.5, markersize=8)
ax1.plot(m_values, fly_mnist, marker='s', label=f'FlyHash (k={int(OPTIMAL_K_RATIO*100)}%)', linewidth=2.5, markersize=8)
ax1.plot(m_values, thresh_mnist, marker='^', label=f'ThresholdHash ($\\theta$={OPTIMAL_THETA})', linewidth=2.5, markersize=8)
ax1.set_title('MNIST: Data with Manifold Structure', fontsize=14)
ax1.set_xlabel('Hash Length ($m$)', fontsize=12)
ax1.set_ylabel('Mean Average Precision (mAP)', fontsize=12)
ax1.set_xticks(m_values)
ax1.legend(fontsize=11)
ax1.grid(True, linestyle='--', alpha=0.7)

# Plot 2: Random Noise (No Manifold Data)
ax2.plot(m_values, lsh_rand, marker='o', label='Dense LSH', linewidth=2.5, markersize=8)
ax2.plot(m_values, fly_rand, marker='s', label=f'FlyHash (k={int(OPTIMAL_K_RATIO*100)}%)', linewidth=2.5, markersize=8)
ax2.plot(m_values, thresh_rand, marker='^', label=f'ThresholdHash ($\\theta$={OPTIMAL_THETA})', linewidth=2.5, markersize=8)
ax2.set_title('Uniform Noise: Data WITHOUT Manifold Structure', fontsize=14)
ax2.set_xlabel('Hash Length ($m$)', fontsize=12)
ax2.set_ylabel('Mean Average Precision (mAP)', fontsize=12)
ax2.set_xticks(m_values)
ax2.legend(fontsize=11)
ax2.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig('manifold_ablation_comparison.png', dpi=300)
print("\nExperiment complete! Plot saved as 'manifold_ablation_comparison.png'")
plt.show()