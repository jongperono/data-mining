import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, r2_score


### -------------------------- ###

# Example 5: SVM for Regression (SVR)

### -------------------------- ###


# 1. Generate non-linear data
np.random.seed(42)
X = np.sort(5 * np.random.rand(100, 1), axis=0)
y = np.sin(X).ravel() + 0.1 * np.random.randn(100)

# 2. Train SVR with RBF kernel
model = make_pipeline(
    StandardScaler(),
    SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.05)
)
model.fit(X, y)

# 3. Predict on a smooth grid
X_plot = np.linspace(0, 5, 200).reshape(-1, 1)
y_pred = model.predict(X_plot)

# 4. Evaluate
y_train_pred = model.predict(X)
print(f"R² score: {r2_score(y, y_train_pred):.4f}")
print(f"MSE: {mean_squared_error(y, y_train_pred):.4f}")

# 5. Plot
plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='darkorange', label='Data')
plt.plot(X_plot, y_pred, color='navy', lw=2, label='SVR (RBF kernel)')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Support Vector Regression')
plt.legend()
plt.show()