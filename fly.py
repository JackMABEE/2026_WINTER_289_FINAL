import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.neighbors import NearestNeighbors
import time

# ==========================================
# 1. Data Loading & Preprocessing
# ==========================================
print("Downloading MNIST dataset (this may take a minute)...")
mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
X = mnist.data

# To keep the experiment fast, we use a subset of the data
# Database (Search space): 10,000 images | Query set: 100 images
DB_SIZE = 10000
QUERY_SIZE = 100
X_db = X[:DB_SIZE]
X_query = X[DB_SIZE:DB_SIZE + QUERY_SIZE]

print("Preprocessing data...")
# Mean-centering and L2 Normalization (Crucial for Cosine/Euclidean equivalence)
mean_vec = np.mean(X_db, axis=0)
X_db = X_db - mean_vec
X_query = X_query - mean_vec

X_db = X_db / (np.linalg.norm(X_db, axis=1, keepdims=True) + 1e-9)
X_query = X_query / (np.linalg.norm(X_query, axis=1, keepdims=True) + 1e-9)

d = X_db.shape[1] # Dimension = 784

# ==========================================
# 2. Compute Ground Truth (Exact Nearest Neighbors)
# ==========================================
print("Computing exact ground truth neighbors...")
# We want to find the top 50 exact Euclidean nearest neighbors for each query
TOP_N = 50 
nbrs = NearestNeighbors(n_neighbors=TOP_N, algorithm='brute', metric='euclidean')
nbrs.fit(X_db)
_, ground_truth_indices = nbrs.kneighbors(X_query)


# ==========================================
# 3. Hashing Algorithms
# ==========================================

def lsh_hash(X, m):
    """Traditional Locality-Sensitive Hashing (Dense + Sign)"""
    np.random.seed(42) # For reproducibility
    W = np.random.randn(X.shape[1], m)
    activations = X @ W
    return (activations > 0).astype(int)

def fly_hash(X, m, k, sparsity=0.1):
    """FlyHash (Sparse Projection + Top-k Winner-Take-All)"""
    np.random.seed(42)
    # 1. Sparse random projection
    W = np.random.randn(X.shape[1], m)
    mask = (np.random.rand(X.shape[1], m) < sparsity).astype(float)
    W = W * mask
    
    activations = X @ W
    
    # 2. Winner-Take-All (WTA)
    binary_hashes = np.zeros_like(activations, dtype=int)
    # Get indices of top k elements for each row
    top_k_indices = np.argsort(activations, axis=1)[:, -k:]
    
    # Set those indices to 1
    rows = np.arange(X.shape[0])[:, None]
    binary_hashes[rows, top_k_indices] = 1
    return binary_hashes

def threshold_hash(X, m, threshold=0.1, sparsity=0.1):
    """Expressivity Paper (Sparse Projection + Constant Threshold)"""
    np.random.seed(42)
    # 1. Sparse random projection (same as Fly)
    W = np.random.randn(X.shape[1], m)
    mask = (np.random.rand(X.shape[1], m) < sparsity).astype(float)
    W = W * mask
    
    activations = X @ W
    
    # 2. Constant Thresholding instead of sorting
    return (activations > threshold).astype(int)


# ==========================================
# 4. Evaluation Function (mAP)
# ==========================================
def compute_map(true_nn, query_hashes, db_hashes):
    """Computes Mean Average Precision by comparing hash similarities"""
    num_queries = true_nn.shape[0]
    
    # Calculate similarity using dot product (counts number of overlapping 1s)
    # shape: (num_queries, db_size)
    similarities = query_hashes @ db_hashes.T 
    
    # Sort database indices by similarity (descending)
    sorted_retrieval = np.argsort(-similarities, axis=1)
    
    aps = []
    for i in range(num_queries):
        truth_set = set(true_nn[i])
        retrieved_list = sorted_retrieval[i]
        
        hits = 0
        sum_precisions = 0.0
        
        # We look at the top 1000 retrieved items to calculate AP
        for rank, doc_id in enumerate(retrieved_list[:1000]):
            if doc_id in truth_set:
                hits += 1
                sum_precisions += hits / (rank + 1.0)
                if hits == len(truth_set):
                    break
        
        if hits > 0:
            aps.append(sum_precisions / len(truth_set))
        else:
            aps.append(0.0)
            
    return np.mean(aps)


# ==========================================
# 5. Experiment Loop
# ==========================================
# Test different expansion dimensions (m)
m_values = [500, 1000, 2000, 4000]

map_lsh = []
map_fly = []
map_threshold = []

print("\nStarting experiment loop...")
for m in m_values:
    print(f"\nEvaluating Expansion Dimension m = {m}")
    start_time = time.time()
    
    # 1. LSH
    db_hash_lsh = lsh_hash(X_db, m)
    q_hash_lsh = lsh_hash(X_query, m)
    score_lsh = compute_map(ground_truth_indices, q_hash_lsh, db_hash_lsh)
    map_lsh.append(score_lsh)
    print(f"  LSH mAP:       {score_lsh:.4f}")
    
    # 2. FlyHash (Sparsity k is 5% of m)
    k = max(1, int(0.05 * m))
    db_hash_fly = fly_hash(X_db, m, k)
    q_hash_fly = fly_hash(X_query, m, k)
    score_fly = compute_map(ground_truth_indices, q_hash_fly, db_hash_fly)
    map_fly.append(score_fly)
    print(f"  FlyHash mAP:   {score_fly:.4f}")
    
    # 3. ThresholdHash (Adaptive Manifold)
    db_hash_thresh = threshold_hash(X_db, m, threshold=0.1)
    q_hash_thresh = threshold_hash(X_query, m, threshold=0.1)
    score_thresh = compute_map(ground_truth_indices, q_hash_thresh, db_hash_thresh)
    map_threshold.append(score_thresh)
    print(f"  Threshold mAP: {score_thresh:.4f}")
    
    print(f"  Time taken: {time.time() - start_time:.2f} seconds")

# ==========================================
# 6. Plotting the Results
# ==========================================
plt.figure(figsize=(8, 6))
plt.plot(m_values, map_lsh, marker='o', linestyle='-', linewidth=2, label='Dense LSH (Sign)')
plt.plot(m_values, map_fly, marker='s', linestyle='-', linewidth=2, label='FlyHash (Top-K WTA)')
plt.plot(m_values, map_threshold, marker='^', linestyle='-', linewidth=2, label='ThresholdHash (Manifold Adaptive)')

plt.title('Approximate Nearest Neighbor Search on MNIST', fontsize=14)
plt.xlabel('Hash Length / Expansion Dimension ($m$)', fontsize=12)
plt.ylabel('Mean Average Precision (mAP)', fontsize=12)
plt.xticks(m_values)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(fontsize=12)
plt.tight_layout()

# Save the plot to a file so you can upload it to your report
plt.savefig('hash_comparison_results.png', dpi=300)
print("\nExperiment complete! Plot saved as 'hash_comparison_results.png'")
plt.show()