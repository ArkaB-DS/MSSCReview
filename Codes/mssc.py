import numpy as np
from sklearn.neighbors import KernelDensity
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.preprocessing import normalize
from scipy.spatial.distance import pdist, squareform
import networkx as nx
from tqdm import tqdm

# --- Step 1: Mean Shift via KDE ---
def mean_shift(X, bandwidth=1.0, max_iter=300, tol=1e-3):
    n, d = X.shape
    shifted = np.copy(X)
    
    for it in tqdm(range(max_iter), desc="Mean Shift Iterations"):
        dist = squareform(pdist(shifted))  # (n, n)

        if np.isscalar(bandwidth):
            kernel_vals = np.exp(-dist**2 / (2 * bandwidth**2))
        else:
            # Variable bandwidths: bandwidth[i] used for the i-th point as denominator
            bw_matrix = np.outer(bandwidth, bandwidth)
            kernel_vals = np.exp(-dist**2 / (2 * bw_matrix**2 + 1e-8))

        weights = kernel_vals / kernel_vals.sum(axis=1, keepdims=True)
        new_shifted = weights @ X
        if np.linalg.norm(new_shifted - shifted) < tol:
            break
        shifted = new_shifted
    return shifted

# --- Step 2: Cluster converged modes ---
def cluster_modes(shifted_pts, tol=1e-2):
    modes = []
    labels = np.full(len(shifted_pts), -1)
    cluster_id = 0
    
    for i, pt in enumerate(tqdm(shifted_pts, desc="Clustering Modes")):
        for j, mode in enumerate(modes):
            if np.linalg.norm(pt - mode) < tol:
                labels[i] = j
                break
        if labels[i] == -1:
            modes.append(pt)
            labels[i] = cluster_id
            cluster_id += 1
    return np.array(modes), labels

# --- Step 3: KDE for each cluster ---
def estimate_partition_kdes(X, labels, bandwidth=1.0):
    unique_labels = np.unique(labels)
    kdes = []

    for l in tqdm(unique_labels, desc="Estimating KDEs"):
        cluster_points = X[labels == l]

        if np.isscalar(bandwidth):
            bw = bandwidth
        else:
            bw = np.mean(bandwidth[labels == l])  # average per-cluster bandwidth

        kde = KernelDensity(kernel='gaussian', bandwidth=bw)
        kde.fit(cluster_points)
        kdes.append(kde)
    return kdes

# --- Step 4: Affinity matrix from KDEs ---
def compute_kde_affinity(kdes, sample_grid):
    K = len(kdes)
    P = np.zeros((K, len(sample_grid)))

    for i, kde in tqdm(enumerate(kdes), desc="Computing Affinity Matrix", total=K):
        P[i] = np.exp(kde.score_samples(sample_grid))
    
    P = normalize(P, norm='l2', axis=1)
    return P @ P.T

# --- Step 5 & 6: Spectral Edge Pruning ---
def spectral_edge_pruning(Ktilde, target_k=None, threshold=None):
    G = nx.Graph()
    K = Ktilde.shape[0]

    for i in tqdm(range(K), desc="Building Graph"):
        for j in range(i+1, K):
            G.add_edge(i, j, weight=Ktilde[i, j])

    edges = sorted(G.edges(data=True), key=lambda e: e[2]['weight'])

    for u, v, w in edges:
        G.remove_edge(u, v)
        if target_k and nx.number_connected_components(G) == target_k:
            break
        if threshold and w['weight'] > threshold:
            G.add_edge(u, v, weight=w['weight'])  # re-add
            break

    components = list(nx.connected_components(G))
    mode_labels = np.zeros(K, dtype=int)
    for idx, comp in enumerate(components):
        for i in comp:
            mode_labels[i] = idx
    return mode_labels

# --- MSSC Wrapper ---
def mean_shift_spectral_clustering(X, bandwidth=1.0, target_k=None, sample_grid=None):
    shifted = mean_shift(X, bandwidth=bandwidth)
    modes, point_labels = cluster_modes(shifted)
    kdes = estimate_partition_kdes(X, point_labels, bandwidth=bandwidth)

    if sample_grid is None:
        sample_grid = X

    Ktilde = compute_kde_affinity(kdes, sample_grid)
    mode_labels = spectral_edge_pruning(Ktilde, target_k=target_k)

    final_labels = np.array([mode_labels[label] for label in point_labels])
    return final_labels, Ktilde
