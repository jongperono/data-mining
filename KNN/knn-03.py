import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


### -------------------------- ###

# Example 3: Finding the Best K (Elbow Method)

### -------------------------- ###


# 1. Load
iris = load_iris()
X, y = iris.data, iris.target

# 2. Try different K values
k_values = range(1, 31)
cv_scores = []

for k in k_values:
    knn = make_pipeline(
        StandardScaler(),
        KNeighborsClassifier(n_neighbors=k)
    )
    scores = cross_val_score(knn, X, y, cv=5, scoring='accuracy')
    cv_scores.append(scores.mean())

# 3. Find best K
best_k = k_values[cv_scores.index(max(cv_scores))]
print(f"Best K: {best_k} (accuracy: {max(cv_scores):.4f})")

# 4. Plot
plt.figure(figsize=(10, 6))
plt.plot(k_values, cv_scores, marker='o', linestyle='-', color='steelblue')
plt.axvline(best_k, color='red', linestyle='--', label=f'Best K = {best_k}')
plt.xlabel('K (number of neighbors)')
plt.ylabel('Cross-Validation Accuracy')
plt.title('Choosing the Best K')
plt.legend()
plt.grid(True)
plt.show()