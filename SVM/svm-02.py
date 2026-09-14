import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score


### -------------------------- ###

# Example 2: SVM with Scaling + Visualization (2D)

### -------------------------- ###


# 1. Load only 2 features for easy plotting
iris = datasets.load_iris()
X = iris.data[:, :2]   # sepal length & width
y = iris.target

# 2. Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# 3. Build pipeline: scale → SVM
model = make_pipeline(
    StandardScaler(),
    SVC(kernel='linear', C=1.0)
)

# 4. Train
model.fit(X_train, y_train)

# 5. Predict & evaluate
y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")

# 6. Visualize decision boundary
x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))

Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.figure(figsize=(10, 6))
plt.contourf(xx, yy, Z, alpha=0.3, cmap='viridis')
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', edgecolors='k', s=50)
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.title('SVM with Linear Kernel')
plt.show()