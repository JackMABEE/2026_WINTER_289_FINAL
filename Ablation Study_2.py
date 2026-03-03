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
# 2. Compute Ground Truth Exact Neighbors
# ==========================================
print("Computing exact ground truth neighbors...")
TOP_N = 50 
nbrs = NearestNeighbors(n_neighbors=TOP_N, algorithm='brute', metric='euclidean')
nbrs.fit(X_db)
_, ground_truth_indices = nbrs.kneighbors(X_query)

# ==========================================
# 3. Core Hashing Functions
# ==========================================
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
# 4. Parameter Ablation Study (UPDATED RANGE)
# ==========================================
M_FIXED = 2000  

# --- Experiment A: FlyHash Sparsity (Expanded Range) ---
# Testing k from 5% all the way to 80% to find the drop-off
k_percentages = [0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.80]
fly_maps = []

print(f"\n[Experiment A] Tuning FlyHash Sparsity (m={M_FIXED})...")
for p in k_percentages:
    k = max(1, int(M_FIXED * p))
    db_hash = fly_hash(X_db, M_FIXED, k)
    q_hash = fly_hash(X_query, M_FIXED, k)
    score = compute_map(ground_truth_indices, q_hash, db_hash)
    fly_maps.append(score)
    print(f"  k = {p*100:2.0f}% (k={k:4d}) | mAP: {score:.4f}")

# --- Experiment B: ThresholdHash Theta (Zoomed In) ---
# Zooming in on the 0.0 to 0.4 range since values > 0.5 kill the activations
thresholds = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35]
thresh_maps = []

print(f"\n[Experiment B] Tuning ThresholdHash Theta (m={M_FIXED})...")
for t in thresholds:
    db_hash = threshold_hash(X_db, M_FIXED, threshold=t)
    q_hash = threshold_hash(X_query, M_FIXED, threshold=t)
    score = compute_map(ground_truth_indices, q_hash, db_hash)
    thresh_maps.append(score)
    print(f"  Theta = {t:4.2f} | mAP: {score:.4f}")

# ==========================================
# 5. Plotting Side-by-Side Results
# ==========================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

k_labels = [f"{int(p*100)}%" for p in k_percentages]
ax1.plot(k_labels, fly_maps, marker='s', color='tab:orange', linestyle='-', linewidth=2.5, markersize=8)
ax1.set_title('FlyHash: Impact of Sparsity ($k$)', fontsize=14)
ax1.set_xlabel('Active Neurons Percentage ($k / m$)', fontsize=12)
ax1.set_ylabel('Mean Average Precision (mAP)', fontsize=12)
ax1.grid(True, linestyle='--', alpha=0.7)

ax2.plot(thresholds, thresh_maps, marker='^', color='tab:green', linestyle='-', linewidth=2.5, markersize=8)
ax2.set_title('ThresholdHash: Impact of Activation Threshold ($\\theta$)', fontsize=14)
ax2.set_xlabel('Threshold ($\\theta$)', fontsize=12)
ax2.set_ylabel('Mean Average Precision (mAP)', fontsize=12)
ax2.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig('ablation_study_optimal.png', dpi=300)
print("\nExperiment complete! Plot saved as 'ablation_study_optimal.png'")
plt.show()