import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


### -------------------------- ###

# Example 4: Visualizing KNN Decision Boundaries

### -------------------------- ###


# 1. Load only 2 features
iris = load_iris()
X = iris.data[:, :2]
y = iris.target

# 2. Compare different K values
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for ax, k in zip(axes, [1, 5, 50]):
    # Train
    knn = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=k))
    knn.fit(X, y)
    
    # Mesh grid
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))
    
    Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Plot
    ax.contourf(xx, yy, Z, alpha=0.3, cmap='viridis')
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', 
               edgecolors='k', s=30)
    ax.set_title(f'K = {k}')
    ax.set_xlabel('Sepal length')
    ax.set_ylabel('Sepal width')

plt.tight_layout()
plt.show()