import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.neighbors import NearestNeighbors
import time

# ==========================================
# 1. Data Loading & Preprocessing
# ==========================================
print("Downloading MNIST dataset...")
mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='liac-arff')
X = mnist.data

DB_SIZE = 10000
QUERY_SIZE = 100
X_db = X[:DB_SIZE]
X_query = X[DB_SIZE:DB_SIZE + QUERY_SIZE]

print("Preprocessing data...")
mean_vec = np.mean(X_db, axis=0)
X_db = X_db - mean_vec
X_query = X_query - mean_vec
X_db = X_db / (np.linalg.norm(X_db, axis=1, keepdims=True) + 1e-9)
X_query = X_query / (np.linalg.norm(X_query, axis=1, keepdims=True) + 1e-9)

# ==========================================
# 2. Compute Exact Nearest Neighbors
# ==========================================
print("Computing exact ground truth neighbors...")
TOP_N = 50 
nbrs = NearestNeighbors(n_neighbors=TOP_N, algorithm='brute', metric='euclidean')
nbrs.fit(X_db)
_, ground_truth_indices = nbrs.kneighbors(X_query)

# ==========================================
# 3. Hashing Functions
# ==========================================
def lsh_hash(X, m):
    """Traditional Dense Random Projection with Sign Hash"""
    np.random.seed(42)
    W = np.random.randn(X.shape[1], m)
    return (X @ W > 0).astype(int)

def fly_hash(X, m, k, sparsity=0.1):
    """Bio-inspired FlyHash with Top-k WTA"""
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
    """Manifold Adaptive Hash with Fixed Threshold"""
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
# 4. Final Evaluation with Optimal Parameters
# ==========================================
m_values = [500, 1000, 2000, 4000]

# Optimal parameters found from our ablation study!
OPTIMAL_K_RATIO = 0.45
OPTIMAL_THETA = 0.05

lsh_maps = []
fly_maps = []
thresh_maps = []

print("\nRunning final comparison with optimal hyperparameters...")
for m in m_values:
    print(f"\nEvaluating m = {m}")
    
    # 1. Dense LSH
    db_lsh = lsh_hash(X_db, m)
    q_lsh = lsh_hash(X_query, m)
    map_lsh = compute_map(ground_truth_indices, q_lsh, db_lsh)
    lsh_maps.append(map_lsh)
    print(f"  [LSH]          mAP: {map_lsh:.4f}")
    
    # 2. FlyHash (using dynamic optimal k)
    optimal_k = max(1, int(m * OPTIMAL_K_RATIO))
    db_fly = fly_hash(X_db, m, optimal_k)
    q_fly = fly_hash(X_query, m, optimal_k)
    map_fly = compute_map(ground_truth_indices, q_fly, db_fly)
    fly_maps.append(map_fly)
    print(f"  [FlyHash]      mAP: {map_fly:.4f} (using k={optimal_k})")
    
    # 3. ThresholdHash (using optimal theta)
    db_thresh = threshold_hash(X_db, m, OPTIMAL_THETA)
    q_thresh = threshold_hash(X_query, m, OPTIMAL_THETA)
    map_thresh = compute_map(ground_truth_indices, q_thresh, db_thresh)
    thresh_maps.append(map_thresh)
    print(f"  [Threshold]    mAP: {map_thresh:.4f} (using theta={OPTIMAL_THETA})")

# ==========================================
# 5. Plotting the Final Results
# ==========================================
plt.figure(figsize=(10, 7))

plt.plot(m_values, lsh_maps, marker='o', label='Dense LSH (Baseline)', linewidth=2.5, markersize=8)
plt.plot(m_values, fly_maps, marker='s', label=f'FlyHash (Optimized, $k={int(OPTIMAL_K_RATIO*100)}\%$)', linewidth=2.5, markersize=8)
plt.plot(m_values, thresh_maps, marker='^', label=f'ThresholdHash (Optimized, $\\theta={OPTIMAL_THETA}$)', linewidth=2.5, markersize=8)

plt.title('Final Performance: Approximate Nearest Neighbor Search on MNIST', fontsize=16)
plt.xlabel('Hash Length / Expansion Dimension ($m$)', fontsize=14)
plt.ylabel('Mean Average Precision (mAP)', fontsize=14)
plt.xticks(m_values)
plt.legend(fontsize=12, loc='lower right')
plt.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig('final_optimized_comparison.png', dpi=300)
print("\nFinal plot saved as 'final_optimized_comparison.png'!")
plt.show()