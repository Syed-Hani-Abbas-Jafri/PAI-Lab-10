import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

wine = load_wine()
X, y = wine.data, wine.target
print("Number of samples :", X.shape[0])
print("Number of features:", X.shape[1])
print("Number of classes :", len(wine.target_names), list(wine.target_names))
print("Feature names     :")
for name in wine.feature_names:
    print("   -", name)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

models = {
    "KNN (k=5)": KNeighborsClassifier(n_neighbors=5),
    "Decision Tree (depth=4)": DecisionTreeClassifier(max_depth=4, random_state=42),
    "Random Forest (100)": RandomForestClassifier(n_estimators=100, random_state=42),
}
print("\n" + "-" * 40)
print(f"{'Model':<26}{'Accuracy':>10}")
print("-" * 40)
for name, model in models.items():
    model.fit(X_train, y_train)
    print(f"{name:<26}{model.score(X_test, y_test):>10.4f}")
print("-" * 40)

ks = range(1, 21)
accs = [KNeighborsClassifier(n_neighbors=k).fit(X_train, y_train).score(X_test, y_test)
        for k in ks]
best_k = ks[int(np.argmax(accs))]
print(f"\nBest k = {best_k} with accuracy = {max(accs):.4f}")

plt.plot(ks, accs, marker="o")
plt.axvline(best_k, color="red", linestyle="--", label=f"best k = {best_k}")
plt.xticks(list(ks))
plt.xlabel("k (number of neighbors)")
plt.ylabel("Test accuracy")
plt.title("KNN accuracy vs k (Wine dataset)")
plt.legend()
plt.grid(alpha=0.3)
plt.show()
