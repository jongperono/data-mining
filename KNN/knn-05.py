from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score


### -------------------------- ###

# Example 5: Weighted KNN (Distance-Based Voting)

### -------------------------- ###


# 1. Load data
data = load_breast_cancer()
X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# 2. Compare uniform vs distance weighting
for weights in ['uniform', 'distance']:
    knn = make_pipeline(
        StandardScaler(),
        KNeighborsClassifier(n_neighbors=5, weights=weights)
    )
    knn.fit(X_train, y_train)
    acc = accuracy_score(y_test, knn.predict(X_test))
    print(f"weights='{weights}' → Accuracy: {acc:.4f}")