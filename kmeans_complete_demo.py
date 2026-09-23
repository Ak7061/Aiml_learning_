"""
K-MEANS COMPLETE DEMO — matches "KMeans_Complete_Notes.docx"
==============================================================
Covers, in order:
  1. From-scratch K-Means (so the assign -> update loop is visible in code)
  2. Same 9-point dataset used in the notes, K = 3
  3. K-Means++ initialization implemented from scratch (compare vs random init)
  4. scikit-learn KMeans (production version, confirms our from-scratch result)
  5. Elbow Method (WCSS vs K)
  6. Silhouette Method (score vs K)

Run:  python kmeans_complete_demo.py
Produces: kmeans_steps.png, kmeans_pp_vs_random.png, elbow_silhouette.png
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples

np.random.seed(42)

# ---------------------------------------------------------------------------
# 1. THE DATASET (identical to the notes, Section 2)
# ---------------------------------------------------------------------------
X = np.array([
    [1.0, 1.0],   # P1
    [1.5, 2.0],   # P2
    [2.0, 1.2],   # P3
    [4.0, 5.0],   # P4
    [4.5, 5.5],   # P5
    [5.0, 4.5],   # P6
    [8.0, 1.0],   # P7
    [9.0, 2.0],   # P8
    [8.5, 1.5],   # P9
])
labels_points = [f"P{i+1}" for i in range(len(X))]


# ---------------------------------------------------------------------------
# 2. FROM-SCRATCH K-MEANS  (Steps 3-7 of the notes, written as code)
# ---------------------------------------------------------------------------
def kmeans_from_scratch(X, initial_centroids, max_iter=100, verbose=True):
    """
    Implements exactly the loop described in the notes:
      Step 3: distance    Step 4: assign
      Step 5: update       Step 6-7: repeat until convergence
    Returns: final centroids, final labels, history of centroids (for plotting)
    """
    centroids = np.array(initial_centroids, dtype=float)
    history = [centroids.copy()]

    for iteration in range(max_iter):
        # --- Step 3 & 4: distance + assignment ---
        # distances[i, k] = distance from point i to centroid k
        distances = np.linalg.norm(X[:, None, :] - centroids[None, :, :], axis=2)
        assignments = np.argmin(distances, axis=1)  # nearest centroid per point

        # --- Step 5: update centroids to the mean of their assigned points ---
        new_centroids = np.array([
            X[assignments == k].mean(axis=0) if np.any(assignments == k) else centroids[k]
            for k in range(len(centroids))
        ])

        if verbose:
            print(f"Iteration {iteration + 1}: centroids = {np.round(new_centroids, 3).tolist()}")

        history.append(new_centroids.copy())

        # --- Step 7: converge when centroids stop moving ---
        if np.allclose(new_centroids, centroids):
            if verbose:
                print(f"Converged after {iteration + 1} iterations.\n")
            break
        centroids = new_centroids

    return centroids, assignments, history


# Use the same starting centroids as the notes (Section 4, Step 2 example)
initial = [[1, 1], [4, 5], [8, 1]]   # C1, C2, C3 from the notes
print("=" * 60)
print("FROM-SCRATCH K-MEANS  (K=3, same starting centroids as notes)")
print("=" * 60)
final_centroids, final_labels, history = kmeans_from_scratch(X, initial)

print("Final cluster assignments:")
for name, lab in zip(labels_points, final_labels):
    print(f"  {name} -> Cluster {lab + 1}")
print("Final centroids:", np.round(final_centroids, 3).tolist())

# Sanity check against the notes' manual example (Section 6):
# Cluster 1 should contain P1,P2,P3 with mean (1.5, 1.4)
print("\nCheck against notes Section 6 (expected mean (1.5, 1.4) for P1,P2,P3):")
print(" Computed:", np.round(X[[0, 1, 2]].mean(axis=0), 3))


# ---------------------------------------------------------------------------
# Plot: iteration-by-iteration movement of centroids
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 6))
colors = ["#e74c3c", "#3498db", "#2ecc71"]
for k in range(3):
    pts = X[final_labels == k]
    ax.scatter(pts[:, 0], pts[:, 1], color=colors[k], s=90, label=f"Cluster {k+1}", zorder=3)

for i, name in enumerate(labels_points):
    ax.annotate(name, (X[i, 0] + 0.1, X[i, 1] + 0.1), fontsize=9)

hist = np.array(history)  # shape (n_iters+1, K, 2)
for k in range(3):
    path = hist[:, k, :]
    ax.plot(path[:, 0], path[:, 1], "--", color=colors[k], alpha=0.6, zorder=2)
    ax.scatter(path[0, 0], path[0, 1], marker="x", s=140, color=colors[k], zorder=4)  # start
    ax.scatter(path[-1, 0], path[-1, 1], marker="*", s=300, color=colors[k],
               edgecolor="black", zorder=5)  # final

ax.set_title("From-scratch K-Means: centroid path (x = start, * = final)")
ax.set_xlabel("x1"); ax.set_ylabel("x2")
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("kmeans_steps.png", dpi=130)
plt.close()


# ---------------------------------------------------------------------------
# 3. K-MEANS++ INITIALIZATION FROM SCRATCH  (notes Section 8)
# ---------------------------------------------------------------------------
def kmeans_plus_plus_init(X, k, rng):
    """
    Notes Section 8, steps 1-5, implemented directly:
      1) first centroid random
      2) D(x) = distance to nearest already-chosen centroid
      3) probability proportional to D(x)^2
      4) sample next centroid using that probability
      5) repeat until k centroids chosen
    """
    n = X.shape[0]
    centroids = [X[rng.integers(n)]]  # Step 1

    for _ in range(1, k):
        dist_sq = np.array([
            min(np.sum((x - c) ** 2) for c in centroids)  # Step 2: D(x)^2
            for x in X
        ])
        probs = dist_sq / dist_sq.sum()                    # Step 3
        next_idx = rng.choice(n, p=probs)                  # Step 4
        centroids.append(X[next_idx])
    return np.array(centroids)                             # Step 5 done


rng = np.random.default_rng(7)
pp_init = kmeans_plus_plus_init(X, 3, rng)
print("\nK-Means++ chosen initial centroids:", np.round(pp_init, 2).tolist())

# Compare bad random init vs K-Means++ init, run scikit-learn's finishing loop on both
bad_random_init = np.array([[1.2, 1.1], [1.6, 1.3], [2.0, 1.0]])  # all 3 near the same corner

km_bad = KMeans(n_clusters=3, init=bad_random_init, n_init=1, random_state=0).fit(X)
km_pp = KMeans(n_clusters=3, init=pp_init, n_init=1, random_state=0).fit(X)

print(f"\nInertia (WCSS) with a poor random init : {km_bad.inertia_:.3f}")
print(f"Inertia (WCSS) with K-Means++ init      : {km_pp.inertia_:.3f}")
print("Note: this toy dataset has 3 very well-separated blobs, so even a poor init")
print("recovers the global optimum here. On messier/overlapping real datasets, a bad")
print("random init more often gets stuck in a worse local optimum -- that's the failure")
print("mode K-Means++ is specifically designed to avoid.")

fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
for ax, model, title in [(axes[0], km_bad, "Poor random init (all 3 centroids clumped)"),
                          (axes[1], km_pp, "K-Means++ init (spread out)")]:
    for k in range(3):
        pts = X[model.labels_ == k]
        ax.scatter(pts[:, 0], pts[:, 1], color=colors[k], s=80)
    ax.scatter(model.cluster_centers_[:, 0], model.cluster_centers_[:, 1],
               marker="*", s=300, color="black")
    ax.set_title(f"{title}\nWCSS = {model.inertia_:.2f}")
    ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("kmeans_pp_vs_random.png", dpi=130)
plt.close()


# ---------------------------------------------------------------------------
# 4. SCIKIT-LEARN K-MEANS  (production version — confirms from-scratch result)
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("SCIKIT-LEARN KMeans (init='k-means++', the library default)")
print("=" * 60)
sk_model = KMeans(n_clusters=3, init="k-means++", n_init=10, random_state=42).fit(X)
print("sklearn labels:   ", sk_model.labels_.tolist())
print("sklearn centroids:", np.round(sk_model.cluster_centers_, 3).tolist())
print("sklearn inertia (WCSS):", round(sk_model.inertia_, 3))


# ---------------------------------------------------------------------------
# 5. ELBOW METHOD  (notes Section 9)
# ---------------------------------------------------------------------------
K_range = range(1, 8)
wcss = []
for k in K_range:
    m = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=42).fit(X)
    wcss.append(m.inertia_)

print("\nElbow method — WCSS per K:")
for k, w in zip(K_range, wcss):
    print(f"  K={k}: WCSS={w:.3f}")


# ---------------------------------------------------------------------------
# 6. SILHOUETTE METHOD  (notes Section 10)
# ---------------------------------------------------------------------------
sil_scores = {}
for k in range(2, 8):  # silhouette needs at least 2 clusters
    m = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=42).fit(X)
    sil_scores[k] = silhouette_score(X, m.labels_)

print("\nSilhouette method — average score per K:")
for k, s in sil_scores.items():
    print(f"  K={k}: silhouette={s:.3f}")

best_k = max(sil_scores, key=sil_scores.get)
print(f"\n-> Silhouette suggests K={best_k} (highest average score)")


# ---------------------------------------------------------------------------
# Plot: Elbow (left) + Silhouette (right) side by side
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].plot(list(K_range), wcss, marker="o", color="#3498db")
axes[0].axvline(3, color="red", linestyle="--", alpha=0.6, label="elbow at K=3")
axes[0].set_title("Elbow Method")
axes[0].set_xlabel("K"); axes[0].set_ylabel("WCSS (inertia)")
axes[0].legend(); axes[0].grid(alpha=0.3)

ks = list(sil_scores.keys())
scores = list(sil_scores.values())
axes[1].plot(ks, scores, marker="o", color="#2ecc71")
axes[1].axvline(best_k, color="red", linestyle="--", alpha=0.6, label=f"best K={best_k}")
axes[1].set_title("Silhouette Method")
axes[1].set_xlabel("K"); axes[1].set_ylabel("Average silhouette score")
axes[1].legend(); axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig("elbow_silhouette.png", dpi=130)
plt.close()

print("\nSaved: kmeans_steps.png, kmeans_pp_vs_random.png, elbow_silhouette.png")
