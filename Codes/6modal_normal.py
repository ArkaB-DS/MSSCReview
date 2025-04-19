import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_blobs
from sklearn.neighbors import NearestNeighbors
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, homogeneity_score
from mssc import mean_shift_spectral_clustering

# --- 1. Generate synthetic data (6-modal mixture) ---
n_samples = 1000
X, y_true = make_blobs(n_samples=n_samples, centers=6, cluster_std=1.5, random_state=42)

# --- 2. Bandwidth using Silverman's Rule ---
def silverman_bandwidth(X):
    n, d = X.shape
    sigma = np.std(X, axis=0).mean()
    h = (4 / (d + 2)) ** (1 / (d + 4)) * sigma * n ** (-1 / (d + 4))
    return h

# --- 3. Variable Bandwidth using k-NN + Gaussian Kernel ---
def spherical_gaussian_kernel(r, bandwidth):
    return np.exp(-0.5 * (r ** 2) / (bandwidth ** 2))

def variable_bandwidth(X, k=5, scaling_factor=1.0):
    nbrs = NearestNeighbors(n_neighbors=k).fit(X)
    distances, _ = nbrs.kneighbors(X)
    mean_distances = np.mean(distances[:, 1:], axis=1)
    bandwidths = np.mean([spherical_gaussian_kernel(d, scaling_factor) for d in mean_distances], axis=0)
    return bandwidths

def optimize_scaling_factor(X):
    gmm = GaussianMixture(n_components=1, covariance_type='full', random_state=42)
    gmm.fit(X)
    return np.sqrt(gmm.covariances_.flatten()[0])

# --- 4. Apply MSSC (Fixed + Variable Bandwidths) ---
fixed_bandwidth = silverman_bandwidth(X)
labels_fixed, _ = mean_shift_spectral_clustering(X, bandwidth=fixed_bandwidth)

scaling_factor = optimize_scaling_factor(X)
bandwidths_variable = variable_bandwidth(X, scaling_factor=scaling_factor)
labels_variable, _ = mean_shift_spectral_clustering(X, bandwidth=bandwidths_variable)

# --- 5. Visualize Results ---
palette = sns.color_palette("tab10", max(len(np.unique(labels_variable)), len(np.unique(labels_fixed))))

fig, axes = plt.subplots(1, 3, figsize=(21, 6))
titles = [
    "True Labels",
    f"MSSC (Fixed Bandwidth = {fixed_bandwidth:.2f})",
    "MSSC (Variable Bandwidth)"
]
labels = [y_true, labels_fixed, labels_variable]

for ax, title, label_set in zip(axes, titles, labels):
    ax.scatter(X[:, 0], X[:, 1], c=[palette[i] for i in label_set], s=30, alpha=0.8)
    ax.set_title(title)
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    ax.grid(True)

plt.suptitle("MSSC: True Labels vs Fixed and Variable Bandwidths", fontsize=16)
plt.tight_layout()
plt.show()

# --- 6. Evaluation ---
def report_metrics(name, labels):
    ari = adjusted_rand_score(y_true, labels)
    nmi = normalized_mutual_info_score(y_true, labels)
    homogeneity = homogeneity_score(y_true, labels)
    print(f"{name}\n  ARI: {ari:.3f} | NMI: {nmi:.3f} | Homogeneity: {homogeneity:.3f}\n")

report_metrics("MSSC (Fixed Bandwidth)", labels_fixed)
report_metrics("MSSC (Variable Bandwidth)", labels_variable)
