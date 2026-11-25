from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, classification_report

def confusion(y_true, y_pred):
    TP = TN = FP = FN = 0
    for t, p in zip(y_true, y_pred):
        if t == 1 and p == 1:
            TP += 1
        elif t == 0 and p == 0:
            TN += 1
        elif t == 0 and p == 1:
            FP += 1
        else:
            FN += 1
    return TP, TN, FP, FN

def precision(y_true, y_pred):
    TP, TN, FP, FN = confusion(y_true, y_pred)
    return TP / (TP + FP) if (TP + FP) else 0.0

def recall(y_true, y_pred):
    TP, TN, FP, FN = confusion(y_true, y_pred)
    return TP / (TP + FN) if (TP + FN) else 0.0

def f1(y_true, y_pred):
    p, r = precision(y_true, y_pred), recall(y_true, y_pred)
    return 2 * p * r / (p + r) if (p + r) else 0.0

def accuracy(y_true, y_pred):
    TP, TN, FP, FN = confusion(y_true, y_pred)
    return (TP + TN) / (TP + TN + FP + FN)

data = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.2, random_state=42, stratify=data.target)

tree = DecisionTreeClassifier(random_state=42).fit(X_train, y_train)
y_pred = tree.predict(X_test)

TP, TN, FP, FN = confusion(y_test, y_pred)
print("TP =", TP, " TN =", TN, " FP =", FP, " FN =", FN)
print("Accuracy  :", round(accuracy(y_test, y_pred), 4))
print("Precision :", round(precision(y_test, y_pred), 4))
print("Recall    :", round(recall(y_test, y_pred), 4))
print("F1-score  :", round(f1(y_test, y_pred), 4))

print("\nsklearn confusion matrix (rows = actual, cols = predicted; order 0,1):")
print(confusion_matrix(y_test, y_pred))
print("\nsklearn classification report:")
print(classification_report(y_test, y_pred, target_names=data.target_names))

print("""Which metric matters most for cancer detection?
Recall (sensitivity) is the most important metric. A false negative means a
patient who actually has cancer is told they are healthy, so they miss
treatment and the consequences can be fatal. A false positive only leads to
extra tests. We therefore want to find as many true cancer cases as possible,
even if precision drops slightly. (Note: here the label for malignant is 0, so
the recall of the malignant class in the report is the figure to watch.)""")
