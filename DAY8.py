#!/usr/bin/env python
# coding: utf-8

# In[2]:


# iris_logreg_corrected.py
# Logistic Regression on the Iris dataset — corrected & ready to run

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def main():
    # -------------------------
    # Load data
    # -------------------------
    iris = load_iris()
    X = iris.data             # 4 features: sepal length/width, petal length/width
    y = iris.target           # labels: 0, 1, 2
    feature_names = iris.feature_names
    target_names = iris.target_names

    # Optional: dataframe for quick inspection
    df = pd.DataFrame(X, columns=feature_names)
    df['target'] = y
    # print(df.head())

    # -------------------------
    # Train / Test split
    # -------------------------
    # - test_size=0.2 -> 20% test set
    # - stratify=y -> keep class proportions balanced in train & test
    # - random_state=42 -> reproducible split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # -------------------------
    # Feature scaling (recommended)
    # -------------------------
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # -------------------------
    # Train Logistic Regression models (no deprecated 'multi_class' argument)
    # -------------------------
    # solver='lbfgs' is a good default for small datasets
    model_raw = LogisticRegression(max_iter=200, solver='lbfgs', random_state=42)
    model_raw.fit(X_train, y_train)

    model_scaled = LogisticRegression(max_iter=200, solver='lbfgs', random_state=42)
    model_scaled.fit(X_train_scaled, y_train)

    # -------------------------
    # Evaluate models
    # -------------------------
    # Predictions
    y_pred_raw = model_raw.predict(X_test)
    y_pred_scaled = model_scaled.predict(X_test_scaled)

    # Accuracy
    acc_raw = accuracy_score(y_test, y_pred_raw)
    acc_scaled = accuracy_score(y_test, y_pred_scaled)

    print("=== Logistic Regression on Iris (raw features) ===")
    print(f"Accuracy: {acc_raw:.4f}")
    print("\nClassification report:")
    print(classification_report(y_test, y_pred_raw, target_names=target_names))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred_raw))

    print("\n\n=== Logistic Regression on Iris (scaled features) ===")
    print(f"Accuracy: {acc_scaled:.4f}")
    print("\nClassification report:")
    print(classification_report(y_test, y_pred_scaled, target_names=target_names))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred_scaled))

    # -------------------------
    # Predicted probabilities (scaled model)
    # -------------------------
    probs = model_scaled.predict_proba(X_test_scaled)
    print("\nPredicted probabilities for the test samples (first 5 shown):")
    for i in range(min(5, len(X_test))):
        true_label = target_names[y_test[i]]
        probs_rounded = np.round(probs[i], 3)
        print(f"Sample {i} — true: {true_label} — probs: {probs_rounded}")

    # -------------------------
    # Example: change decision threshold for one-vs-rest check (optional)
    # -------------------------
    # For multiclass, changing threshold is more complex. Example below shows how you
    # might pick class = argmax(probabilities) (default behavior), included for clarity.
    pred_by_argmax = np.argmax(probs, axis=1)
    print("\nPredictions by argmax(probabilities) (first 10):", pred_by_argmax[:10])

if __name__ == "__main__":
    main()


# In[4]:


from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import numpy as np

# -------------------------
# Step 1: Load data
# -------------------------
iris = load_iris()
X = iris.data
y = iris.target
target_names = iris.target_names

# -------------------------
# Step 2: Split
# -------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# -------------------------
# Step 3: Scale data
# -------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -------------------------
# Step 4: Train model
# -------------------------
model = LogisticRegression(max_iter=200)
model.fit(X_train_scaled, y_train)

# =====================================================
# PART 1 — Print probabilities for test data
# =====================================================
print("=== Probabilities for first 5 test samples ===")
probabilities = model.predict_proba(X_test_scaled)
print(probabilities[:5])

# =====================================================
# PART 2 — Predict species for custom input
# =====================================================

# Custom flower measurements
custom_data = np.array([[5.1, 3.5, 1.4, 0.0]])

# Scale the custom input
custom_data_scaled = scaler.transform(custom_data)

# Predict class (0, 1, or 2)
predicted_class = model.predict(custom_data_scaled)[0]

# Convert class → species name
predicted_species = target_names[predicted_class]

print("\nCustom input prediction:")
print("Predicted species:", predicted_species)


# In[3]:


import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load digits dataset
digits = load_digits()
X = digits.data
y = digits.target

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Values of k to test
k_values = [1, 3, 5, 7, 11]
accuracies = []

# Train and evaluate KNN for each k
for k in k_values:
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    accuracies.append(acc)
    print(f"K={k}, Accuracy={acc:.4f}")

# Plot accuracy vs K
plt.figure(figsize=(6, 4))
plt.plot(k_values, accuracies, marker='o', linestyle='--')
plt.title("KNN Accuracy vs K")
plt.xlabel("K")
plt.ylabel("Accuracy")
plt.grid(True)
plt.show()

# Visualize digit at index 100
plt.figure(figsize=(3, 3))
plt.imshow(digits.images[100], cmap='gray')
plt.title(f"Digit Label: {digits.target[100]}")
plt.axis('off')
plt.show()


# In[5]:


import numpy as np
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

# -----------------------------
# Load dataset
# -----------------------------
iris = load_iris()
X = iris.data
y = iris.target
target_names = iris.target_names

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# Train KNN model
# -----------------------------
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)

# Predictions
pred = model.predict(X_test)

# -----------------------------
# Metrics
# -----------------------------
cm = confusion_matrix(y_test, pred)
cr = classification_report(y_test, pred, target_names=target_names, output_dict=True)

print("Confusion Matrix:\n")
print(cm)
print("\nClassification Report:\n")
print(classification_report(y_test, pred, target_names=target_names))

# -----------------------------
# Identify best precision species
# -----------------------------
precision_scores = {cls: cr[cls]["precision"] for cls in target_names}
best_precision_species = max(precision_scores, key=precision_scores.get)

# -----------------------------
# Identify lowest recall species
# -----------------------------
recall_scores = {cls: cr[cls]["recall"] for cls in target_names}
lowest_recall_species = min(recall_scores, key=recall_scores.get)

# -----------------------------
# Print results
# -----------------------------
print("Species with BEST precision:", best_precision_species)
print("Species with LOWEST recall:", lowest_recall_species)

# -----------------------------
# Explain WHY
# -----------------------------
print("\nExplanation:")
print("- Setosa is usually perfectly separable from the others, so it often gets perfect precision and recall.")
print("- Versicolor and Virginica have overlapping feature values, so the model may confuse them.")
print("- This overlap causes misclassifications for Virginica or Versicolor, reducing recall on one of them.")


# In[7]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

# -------------------------------------------
# 1. CREATE A SMALL DATASET
# -------------------------------------------
data = {
    "Income": [30, 45, 55, 25, 70, 85, 40, 60, 20, 90, 50, 33],
    "Age": [25, 35, 45, 22, 50, 60, 30, 40, 21, 55, 38, 28],
    "CreditScore": [580, 650, 720, 540, 780, 810, 600, 700, 500, 790, 640, 590],
    "ExistingLoans": [2, 1, 0, 3, 0, 1, 2, 1, 4, 0, 1, 2],
    "EmploymentStability": [1, 3, 8, 0.5, 10, 12, 2, 7, 0.3, 11, 4, 1.5],
    "LoanStatus": [0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0]   # 1 = repay, 0 = default
}

df = pd.DataFrame(data)
print("Dataset:\n", df)

# -------------------------------------------
# 2. PREPARE DATA (FEATURES + TARGET)
# -------------------------------------------
X = df.drop("LoanStatus", axis=1)
y = df["LoanStatus"]

# -------------------------------------------
# 3. SCALE THE FEATURES (IMPORTANT!)
# -------------------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -------------------------------------------
# 4. TRAIN-TEST SPLIT
# -------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42
)

# -------------------------------------------
# 5. APPLY LOGISTIC REGRESSION
# -------------------------------------------
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Predictions
pred = model.predict(X_test)

# -------------------------------------------
# 6. EVALUATE RESULTS
# -------------------------------------------
print("\nAccuracy:", accuracy_score(y_test, pred))
print("\nClassification Report:\n", classification_report(y_test, pred))

# -------------------------------------------
# 7. FEATURE IMPACT (COEFFICIENTS)
# -------------------------------------------
coefficients = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_[0]
})

coefficients["AbsoluteImpact"] = coefficients["Coefficient"].abs()
coefficients = coefficients.sort_values(by="AbsoluteImpact", ascending=False)

print("\nFeature Impact (higher = stronger influence):\n")
print(coefficients)

strongest_feature = coefficients.iloc[0]["Feature"]
print(f"\nStrongest Impact Feature: {strongest_feature}")


# In[8]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# -------------------------------------------------
# 1. CREATE A SMALL DATASET
# -------------------------------------------------
data = {
    "SugarLevel": [110, 150, 95, 180, 200, 130, 85, 175, 160, 92, 140, 155],
    "HeartRate":  [72, 88, 65, 95, 102, 78, 60, 98, 90, 66, 85, 92],
    "Age":        [25, 45, 32, 55, 60, 40, 28, 50, 48, 30, 42, 53],
    "BMI":        [22.5, 28.3, 20.1, 32.5, 35.0, 26.1, 19.8, 31.0, 29.5, 21.3, 27.7, 30.2],

    # TARGET: 1 = Disease, 0 = Healthy
    "Disease":    [0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1]
}

df = pd.DataFrame(data)
print("Dataset:\n", df)

# -------------------------------------------------
# 2. SPLIT FEATURES AND TARGET
# -------------------------------------------------
X = df.drop("Disease", axis=1)
y = df["Disease"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# -------------------------------------------------
# 3. TEST K VALUES (3,5,7)
# -------------------------------------------------
k_values = [3, 5, 7]
accuracies = {}

for k in k_values:
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    acc = accuracy_score(y_test, pred)
    accuracies[k] = acc
    print(f"K={k} → Accuracy = {acc:.4f}")

# -------------------------------------------------
# 4. IDENTIFY BEST K
# -------------------------------------------------
best_k = max(accuracies, key=accuracies.get)
print("\nBest K:", best_k)
print(f"Highest Accuracy = {accuracies[best_k]:.4f}")


# In[9]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report

# ---------------------------------------------------------
# 1. CREATE SAMPLE DATASET
# ---------------------------------------------------------
data = {
    "OnlineVisits":       [2, 5, 3, 10, 12, 4, 6, 8, 1, 11, 7, 9],
    "TimeOnPage":         [30, 80, 45, 120, 150, 50, 70, 95, 20, 140, 65, 110],   # seconds
    "Clicks":             [1, 3, 2, 5, 7, 2, 3, 4, 1, 6, 3, 5],
    "CartInteractions":   [0, 1, 0, 2, 3, 0, 1, 1, 0, 2, 1, 2],

    # Target: 1 = Bought the product, 0 = Did not buy
    "Bought":             [0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1]
}

df = pd.DataFrame(data)
print("Dataset:\n", df)

# ---------------------------------------------------------
# 2. SPLIT FEATURES AND TARGET
# ---------------------------------------------------------
X = df.drop("Bought", axis=1)
y = df["Bought"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# ---------------------------------------------------------
# 3. TRAIN LOGISTIC REGRESSION
# ---------------------------------------------------------
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Predictions
pred = model.predict(X_test)

# ---------------------------------------------------------
# 4. CONFUSION MATRIX + REPORT
# ---------------------------------------------------------
cm = confusion_matrix(y_test, pred)
print("\nConfusion Matrix:\n", cm)

print("\nClassification Report:\n", classification_report(y_test, pred))


# In[10]:


import numpy as np

# ---------------------------------------------
# 1. SIGMOID FUNCTION
# ---------------------------------------------
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# ---------------------------------------------
# 2. PREDICTION FUNCTION
# ---------------------------------------------
def predict(X, weights):
    z = np.dot(X, weights)
    return sigmoid(z)

# ---------------------------------------------
# 3. GRADIENT DESCENT TRAINING FUNCTION
# ---------------------------------------------
def train_logistic_regression(X, y, lr=0.01, epochs=1000):
    # Add bias term (column of ones)
    X = np.c_[np.ones((X.shape[0], 1)), X]  # shape: (n_samples, n_features+1)

    # Initialize weights
    weights = np.zeros(X.shape[1])

    for epoch in range(epochs):
        # Linear combination
        z = np.dot(X, weights)
        y_pred = sigmoid(z)

        # Gradient calculation
        gradient = np.dot(X.T, (y_pred - y)) / y.size

        # Update weights
        weights -= lr * gradient

    return weights

# ---------------------------------------------
# 4. PREDICTION (CLASS LABEL)
# ---------------------------------------------
def predict_labels(X, weights, threshold=0.5):
    X = np.c_[np.ones((X.shape[0], 1)), X]  # add bias term
    probs = sigmoid(np.dot(X, weights))
    return (probs >= threshold).astype(int)

# ---------------------------------------------
# 5. TEST WITH A SIMPLE DATASET
# ---------------------------------------------
# Example dataset: AND logic gate
X_train = np.array([[0,0],[0,1],[1,0],[1,1]])
y_train = np.array([0, 0, 0, 1])

# Train
weights = train_logistic_regression(X_train, y_train, lr=0.1, epochs=1000)
print("Learned weights:", weights)

# Predict
y_pred = predict_labels(X_train, weights)
print("Predictions:", y_pred)

# Accuracy
accuracy = np.mean(y_pred == y_train)
print("Accuracy:", accuracy)


# In[ ]:


# -----------------------------
# IMPORT LIBRARIES
# -----------------------------
import numpy as np
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification, load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score

# -----------------------------
# TASK B: KNN FROM SCRATCH
# -----------------------------
def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

def knn_predict(X_train, y_train, X_test, k=3):
    y_pred = []
    for x_test in X_test:
        distances = [euclidean_distance(x_test, x_train) for x_train in X_train]
        k_indices = np.argsort(distances)[:k]
        k_labels = [y_train[i] for i in k_indices]
        most_common = Counter(k_labels).most_common(1)[0][0]
        y_pred.append(most_common)
    return np.array(y_pred)

# -----------------------------
# TASK C: DECISION BOUNDARY VISUALIZATION
# -----------------------------
# Create simple 2D dataset
X_vis, y_vis = make_classification(
    n_samples=200, n_features=2, n_redundant=0, n_clusters_per_class=1, n_classes=2, random_state=42
)

# Logistic Regression boundary
lr = LogisticRegression()
lr.fit(X_vis, y_vis)

xx, yy = np.meshgrid(np.linspace(X_vis[:,0].min()-1, X_vis[:,0].max()+1, 200),
                     np.linspace(X_vis[:,1].min()-1, X_vis[:,1].max()+1, 200))
grid = np.c_[xx.ravel(), yy.ravel()]

# Predict grid using Logistic Regression
probs_lr = lr.predict(grid).reshape(xx.shape)

plt.contourf(xx, yy, probs_lr, alpha=0.3, cmap='coolwarm')
plt.scatter(X_vis[:,0], X_vis[:,1], c=y_vis, cmap='coolwarm', edgecolor='k')
plt.title("Logistic Regression Decision Boundary")
plt.show()

# Optional: Decision boundary using KNN from scratch
k_for_boundary = 5
probs_knn = knn_predict(X_vis, y_vis, grid, k=k_for_boundary).reshape(xx.shape)

plt.contourf(xx, yy, probs_knn, alpha=0.3, cmap='coolwarm')
plt.scatter(X_vis[:,0], X_vis[:,1], c=y_vis, cmap='coolwarm', edgecolor='k')
plt.title(f"KNN (k={k_for_boundary}) Decision Boundary")
plt.show()

# -----------------------------
# TASK D: IRIS MULTI-CLASS CLASSIFICATION
# -----------------------------
iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Logistic Regression
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)

# KNN (sklearn)
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train, y_train)
y_pred_knn = knn_model.predict(X_test)

# Metrics comparison
metrics = pd.DataFrame({
    "Model": ["Logistic Regression", "KNN"],
    "Accuracy": [accuracy_score(y_test, y_pred_lr), accuracy_score(y_test, y_pred_knn)],
    "Precision": [precision_score(y_test, y_pred_lr, average='macro'),
                  precision_score(y_test, y_pred_knn, average='macro')],
    "Recall": [recall_score(y_test, y_pred_lr, average='macro'),
               recall_score(y_test, y_pred_knn, average='macro')]
})

print("\nIris Classification Metrics Comparison:\n")
print(metrics)

# -----------------------------
# 5-LINE COMPARISON
# -----------------------------
comparison = """
1. Logistic Regression works well for linearly separable classes.
2. KNN captures nonlinear patterns better due to neighbor voting.
3. Logistic Regression trains faster than KNN.
4. KNN performance depends on chosen K and feature scaling.
5. Both models achieve high accuracy, but KNN may have slightly higher recall for minority classes.
"""
print("\nModel Comparison Summary:\n", comparison)


# In[ ]:




