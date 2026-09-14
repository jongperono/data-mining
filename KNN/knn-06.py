import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, r2_score


### -------------------------- ###

# Example 6: KNN for Regression

### -------------------------- ###


# 1. Generate non-linear data
np.random.seed(42)
X = np.sort(5 * np.random.rand(150, 1), axis=0)
y = np.sin(X).ravel() + 0.2 * np.random.randn(150)

# 2. Train KNN regressors with different K
X_plot = np.linspace(0, 5, 500).reshape(-1, 1)

plt.figure(figsize=(12, 6))
plt.scatter(X, y, color='darkorange', s=20, 
            label='Data', alpha=0.6, zorder=1)

for k, color in zip([1, 5, 20], ['red', 'green', 'blue']):
    knn = make_pipeline(
        StandardScaler(),
        KNeighborsRegressor(n_neighbors=k)
    )
    knn.fit(X, y)
    y_plot = knn.predict(X_plot)
    plt.plot(X_plot, y_plot, color=color, lw=2, 
             label=f'K={k}', zorder=2)

plt.xlabel('X')
plt.ylabel('y')
plt.title('KNN Regression — Effect of K')
plt.legend()
plt.show()