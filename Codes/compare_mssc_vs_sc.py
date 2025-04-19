import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import SpectralClustering, MeanShift  
from sklearn.datasets import make_moons
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from mssc import mean_shift_spectral_clustering  

# --- Silverman's Rule of Thumb for Bandwidth ---
def silverman_bandwidth(X):
    n, d = X.shape
    # Calculate the standard deviation of the data
    std_dev = np.std(X, axis=0)
    # Silverman's rule of thumb for bandwidth
    bandwidth = (4 / (d + 2))**(1 / 5) * np.mean(std_dev) * n**(-1 / 5)
    return bandwidth

# --- Generate Moons Dataset ---
X, y_true = make_moons(n_samples=300, noise=0.1, random_state=42)

# --- Set Parameters ---
target_k = 2  # For the moons dataset

# --- Calculate Bandwidth using Silverman's Rule of Thumb ---
bandwidth = silverman_bandwidth(X)
print(f"Calculated bandwidth for Moons: {bandwidth:.4f}")

# --- Standard Spectral Clustering ---
sc = SpectralClustering(n_clusters=target_k, affinity='rbf', gamma=bandwidth, random_state=42)
sc_labels = sc.fit_predict(X)

# --- Mean Shift Clustering ---
ms = MeanShift(bandwidth=bandwidth)  # Using sklearn's MeanShift
ms_labels = ms.fit_predict(X)

# --- Mean Shift Spectral Clustering (MSSC) ---
mssc_labels, _ = mean_shift_spectral_clustering(X, bandwidth=bandwidth, target_k=target_k)

# --- Compute Evaluation Metrics ---
ari_sc = adjusted_rand_score(y_true, sc_labels)
ari_ms = adjusted_rand_score(y_true, ms_labels)
ari_mssc = adjusted_rand_score(y_true, mssc_labels)

nmi_sc = normalized_mutual_info_score(y_true, sc_labels)
nmi_ms = normalized_mutual_info_score(y_true, ms_labels)
nmi_mssc = normalized_mutual_info_score(y_true, mssc_labels)

# --- Display Results ---
print("Comparison Results:")
print("Dataset          | ARI (SC) | ARI (MS) | ARI (MSSC) | NMI (SC) | NMI (MS) | NMI (MSSC)")
print("--------------------------------------------------------------------------")
print(f"Moons            | {ari_sc:8.3f} | {ari_ms:8.3f} | {ari_mssc:10.3f} | {nmi_sc:8.3f} | {nmi_ms:8.3f} | {nmi_mssc:8.3f}")

# --- Optional: Visualize the Results ---
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Standard Spectral Clustering
axes[0].scatter(X[:, 0], X[:, 1], c=sc_labels, cmap='viridis')
axes[0].set_title("Spectral Clustering (Moons)")

# Mean Shift Clustering
axes[1].scatter(X[:, 0], X[:, 1], c=ms_labels, cmap='viridis')
axes[1].set_title("Mean Shift (Moons)")

# MSSC
axes[2].scatter(X[:, 0], X[:, 1], c=mssc_labels, cmap='viridis')
axes[2].set_title("MSSC (Moons)")

plt.show()
