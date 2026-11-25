import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier

digits = load_digits()
X, y = digits.data, digits.target
print("Data shape:", X.shape)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

pca2 = PCA(n_components=2)
X_2d = pca2.fit_transform(X_train_s)
print("Variance kept by 2 components:", round(pca2.explained_variance_ratio_.sum(), 3))

plt.figure(figsize=(8, 6))
sc = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y_train, cmap="tab10", s=15)
plt.colorbar(sc, label="Digit")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("Digits dataset reduced to 2 components")
plt.show()

pca_full = PCA().fit(X_train_s)
cum = np.cumsum(pca_full.explained_variance_ratio_)
n95 = int(np.argmax(cum >= 0.95)) + 1
print("Components needed for 95% variance:", n95)

plt.plot(range(1, len(cum) + 1), cum)
plt.axhline(0.95, color="red", linestyle="--")
plt.axvline(n95, color="green", linestyle="--", label=f"{n95} components")
plt.xlabel("Number of components")
plt.ylabel("Cumulative explained variance")
plt.legend()
plt.show()

pca95 = PCA(n_components=0.95)
X_train_p = pca95.fit_transform(X_train_s)
X_test_p = pca95.transform(X_test_s)
print("Reduced shape:", X_train_p.shape)

def run_knn(Xtr, Xte, label, repeats=20):
    knn = KNeighborsClassifier(n_neighbors=5)
    t0 = time.perf_counter()
    knn.fit(Xtr, y_train)
    fit_t = time.perf_counter() - t0
    t0 = time.perf_counter()
    for _ in range(repeats):
        pred = knn.predict(Xte)
    pred_t = (time.perf_counter() - t0) / repeats
    acc = np.mean(pred == y_test)
    return label, Xtr.shape[1], acc, fit_t, pred_t

results = [run_knn(X_train_s, X_test_s, "Original (scaled)"),
           run_knn(X_train_p, X_test_p, "PCA (95% variance)")]

print(f"\n{'Model':<22}{'Features':>9}{'Accuracy':>10}{'Fit (s)':>12}{'Predict (s)':>14}")
for label, nf, acc, ft, pt in results:
    print(f"{label:<22}{nf:>9}{acc:>10.4f}{ft:>12.5f}{pt:>14.5f}")

print(f"""
Observation:
PCA cut the feature count from 64 to {n95} while keeping 95% of the variance.
Accuracy stays about the same (it may even rise slightly, because PCA drops
noisy, low-variance directions), while KNN distance computations involve fewer
features, so prediction is typically faster. Timings on a dataset this small
are tiny and noisy, but the saving grows with larger data. The trade-off is
the extra cost of fitting PCA and the loss of interpretability of the features.""")
