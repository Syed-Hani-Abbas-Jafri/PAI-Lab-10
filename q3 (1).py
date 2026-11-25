import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

X, _ = make_blobs(n_samples=200,
                  centers=[[25, 20], [25, 80], [55, 50], [85, 20], [85, 80]],
                  cluster_std=6, random_state=3)

inertias = []
for k in range(1, 11):
    km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X)
    inertias.append(km.inertia_)
    print(f"k = {k:2d}  inertia = {km.inertia_:10.1f}")

best_k = 5
plt.plot(range(1, 11), inertias, marker="o")
plt.axvline(best_k, color="red", linestyle="--", label=f"elbow at k = {best_k}")
plt.xlabel("Number of clusters (k)")
plt.ylabel("Inertia (WCSS)")
plt.title("Elbow Method")
plt.legend()
plt.grid(alpha=0.3)
plt.show()

kmeans = KMeans(n_clusters=best_k, n_init=10, random_state=42)
labels = kmeans.fit_predict(X)
centers = kmeans.cluster_centers_

plt.scatter(X[:, 0], X[:, 1], c=labels, cmap="viridis", s=25)
plt.scatter(centers[:, 0], centers[:, 1], c="red", marker="X", s=250,
            edgecolor="black", label="Centroids")
plt.xlabel("Annual income (thousands)")
plt.ylabel("Spending score (1-100)")
plt.title(f"Customer Segments (K-Means, k = {best_k})")
plt.legend()
plt.show()

def level(value, low, high):
    return "Low" if value < low else ("High" if value >= high else "Medium")

def name_cluster(income, spending):
    inc = level(income, 40, 70)
    spd = level(spending, 40, 65)
    names = {
        ("Low", "Low"): "Budget savers (low income, low spending)",
        ("Low", "High"): "Impulsive spenders (low income, high spending)",
        ("Medium", "Medium"): "Average customers (medium income, medium spending)",
        ("High", "Low"): "Frugal wealthy (high income, low spending)",
        ("High", "High"): "VIP / premium customers (high income, high spending)",
    }
    return names.get((inc, spd), f"{inc} income, {spd} spending")

print("\nCluster summary")
print("-" * 75)
for i in range(best_k):
    count = int(np.sum(labels == i))
    inc, spd = centers[i]
    print(f"Cluster {i}: {count:3d} customers | centroid = ({inc:.1f}, {spd:.1f}) "
          f"-> {name_cluster(inc, spd)}")
