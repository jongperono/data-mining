from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


### -------------------------- ###

# Example 1: Basic Random Forest (Iris Dataset)

### -------------------------- ###


# 1. Load data
iris = load_iris()
X, y = iris.data, iris.target

# 2. Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# 3. Create Random Forest
rf = RandomForestClassifier(
    n_estimators=100,     # 100 trees
    random_state=42
)

# 4. Train
rf.fit(X_train, y_train)

# 5. Predict
y_pred = rf.predict(X_test)

# 6. Evaluate
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))