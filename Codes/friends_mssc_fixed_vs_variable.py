import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, homogeneity_completeness_v_measure, silhouette_score
from mssc import mean_shift_spectral_clustering  # Ensure this supports per-sample bandwidths
from scipy.optimize import minimize_scalar

# --- Silverman's Rule for Fixed Bandwidth ---
def silverman_bandwidth(X):
    n, d = X.shape
    h = (4 / (d + 2)) ** (1 / (d + 4)) * np.std(X) * n ** (-1 / (d + 4))
    return h

# --- Variable Bandwidth with Global Scaling (Optimized) ---
def variable_bandwidth_scaled(X):
    n, _ = X.shape
    k = int(np.sqrt(n))  # Adaptive number of neighbors
    nbrs = NearestNeighbors(n_neighbors=k).fit(X)
    distances, _ = nbrs.kneighbors(X)
    base_bandwidths = np.mean(distances, axis=1)

    # KDE likelihood to optimize scaling factor
    def neg_log_likelihood(scale):
        scaled_bandwidths = base_bandwidths * scale
        n, d = X.shape
        log_probs = []
        for i in range(n):
            h_i = scaled_bandwidths[i]
            diff = X[i] - X
            norm_sq = np.sum((diff / h_i) ** 2, axis=1)
            kernel_vals = np.exp(-0.5 * norm_sq) / (h_i ** d)
            prob = np.sum(kernel_vals) / (n * (2 * np.pi) ** (d / 2))
            log_probs.append(np.log(prob + 1e-10))
        return -np.mean(log_probs)

    res = minimize_scalar(neg_log_likelihood, bounds=(0.1, 10), method='bounded')
    best_scale = res.x
    return base_bandwidths * best_scale

# --- Image Preprocessing ---
image_path = 'friends.jpeg'  # Update as needed
image = Image.open(image_path).resize((100, 100))
image = np.array(image) / 255.0  # Normalize
X = image.reshape(-1, 3)

# --- Bandwidths ---
fixed_bandwidth = silverman_bandwidth(X)
scaled_variable_bandwidths = variable_bandwidth_scaled(X)

# --- MSSC ---
fixed_labels, _ = mean_shift_spectral_clustering(X, bandwidth=fixed_bandwidth)
variable_labels, _ = mean_shift_spectral_clustering(X, bandwidth=scaled_variable_bandwidths)

# --- Spectral Clustering (SC) and Mean Shift (MS) ---
from sklearn.cluster import SpectralClustering, MeanShift

# Spectral Clustering
sc = SpectralClustering(n_clusters=6, affinity='nearest_neighbors', random_state=42)
sc_labels = sc.fit_predict(X)

# Mean Shift
ms = MeanShift(bandwidth=fixed_bandwidth)
ms_labels = ms.fit_predict(X)

# --- Reshape for Visualization ---
h, w = image.shape[:2]
fixed_labels_image = fixed_labels.reshape(h, w)
variable_labels_image = variable_labels.reshape(h, w)
sc_labels_image = sc_labels.reshape(h, w)
ms_labels_image = ms_labels.reshape(h, w)

# --- Plot All Segmentations ---
fig, axes = plt.subplots(1, 5, figsize=(25, 5))

axes[0].imshow(image)
axes[0].set_title("Original Image")
axes[0].axis('off')

axes[1].imshow(sc_labels_image, cmap='viridis')
axes[1].set_title("Spectral Clustering")
axes[1].axis('off')

axes[2].imshow(ms_labels_image, cmap='viridis')
axes[2].set_title("Mean Shift")
axes[2].axis('off')

axes[3].imshow(fixed_labels_image, cmap='viridis')
axes[3].set_title("MSSC (Fixed Bandwidth)")
axes[3].axis('off')

axes[4].imshow(variable_labels_image, cmap='viridis')
axes[4].set_title("MSSC (Variable Bandwidth)")
axes[4].axis('off')

plt.tight_layout()
plt.subplots_adjust(wspace=0.1)
plt.show()


# --- Performance Metrics ---
# Adjusted Rand Index (ARI)
ari_sc = adjusted_rand_score(fixed_labels, sc_labels)
ari_ms = adjusted_rand_score(fixed_labels, ms_labels)
ari_fixed = adjusted_rand_score(fixed_labels, fixed_labels)
ari_variable = adjusted_rand_score(fixed_labels, variable_labels)

# Normalized Mutual Information (NMI)
nmi_sc = normalized_mutual_info_score(fixed_labels, sc_labels)
nmi_ms = normalized_mutual_info_score(fixed_labels, ms_labels)
nmi_fixed = normalized_mutual_info_score(fixed_labels, fixed_labels)
nmi_variable = normalized_mutual_info_score(fixed_labels, variable_labels)

# Homogeneity, Completeness, and V-Measure
homogeneity_sc, completeness_sc, v_measure_sc = homogeneity_completeness_v_measure(fixed_labels, sc_labels)
homogeneity_ms, completeness_ms, v_measure_ms = homogeneity_completeness_v_measure(fixed_labels, ms_labels)
homogeneity_fixed, completeness_fixed, v_measure_fixed = homogeneity_completeness_v_measure(fixed_labels, fixed_labels)
homogeneity_variable, completeness_variable, v_measure_variable = homogeneity_completeness_v_measure(fixed_labels, variable_labels)

# Silhouette Score
silhouette_sc = silhouette_score(X, sc_labels)
silhouette_ms = silhouette_score(X, ms_labels)
silhouette_fixed = silhouette_score(X, fixed_labels)
silhouette_variable = silhouette_score(X, variable_labels)

# Print the performance metrics
print("\nPerformance Metrics for Spectral Clustering (SC):")
print(f"  ARI: {ari_sc:.3f} | NMI: {nmi_sc:.3f} | Homogeneity: {homogeneity_sc:.3f} | Silhouette Score: {silhouette_sc:.3f}")

print("\nPerformance Metrics for Mean Shift (MS):")
print(f"  ARI: {ari_ms:.3f} | NMI: {nmi_ms:.3f} | Homogeneity: {homogeneity_ms:.3f} | Silhouette Score: {silhouette_ms:.3f}")

print("\nPerformance Metrics for MSSC (Fixed Bandwidth):")
print(f"  ARI: {ari_fixed:.3f} | NMI: {nmi_fixed:.3f} | Homogeneity: {homogeneity_fixed:.3f} | Silhouette Score: {silhouette_fixed:.3f}")

print("\nPerformance Metrics for MSSC (Variable Bandwidth):")
print(f"  ARI: {ari_variable:.3f} | NMI: {nmi_variable:.3f} | Homogeneity: {homogeneity_variable:.3f} | Silhouette Score: {silhouette_variable:.3f}")
