from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np


### -------------------------- ###

# Example 4: SVM for Binary Classification (Two Classes)

### -------------------------- ###


# 1. Generate 2-class data
X, y = make_blobs(n_samples=200, centers=2, 
                  cluster_std=1.5, random_state=42)

# 2. Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# 3. Build & train
model = make_pipeline(
    StandardScaler(),
    SVC(kernel='linear', C=1.0)
)
model.fit(X_train, y_train)

# 4. Predict
y_pred = model.predict(X_test)

# 5. Evaluate
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# 6. Visualize with support vectors
svm = model.named_steps['svc']
scaler = model.named_steps['standardscaler']
X_scaled = scaler.transform(X)

# Plot
plt.figure(figsize=(10, 6))
plt.scatter(X_scaled[y==0, 0], X_scaled[y==0, 1], 
            c='red', label='Class 0', edgecolors='k')
plt.scatter(X_scaled[y==1, 0], X_scaled[y==1, 1], 
            c='blue', label='Class 1', edgecolors='k')

# Highlight support vectors
plt.scatter(svm.support_vectors_[:, 0], 
            svm.support_vectors_[:, 1],
            s=200, facecolors='none', edgecolors='green',
            linewidths=2, label='Support Vectors')

plt.legend()
plt.title('SVM with Support Vectors Highlighted')
plt.xlabel('Feature 1 (scaled)')
plt.ylabel('Feature 2 (scaled)')
plt.show()

print(f"\nNumber of support vectors: {len(svm.support_vectors_)}")