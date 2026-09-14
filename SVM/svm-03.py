from sklearn import datasets
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, classification_report


### -------------------------- ###

# Example 3: SVM with Hyperparameter Tuning

### -------------------------- ###


# 1. Load data
iris = datasets.load_iris()
X, y = iris.data, iris.target

# 2. Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# 3. Pipeline
pipe = make_pipeline(StandardScaler(), SVC())

# 4. Define the "knobs" to try
param_grid = {
    'svc__kernel': ['linear', 'rbf', 'poly'],
    'svc__C': [0.1, 1, 10, 100],
    'svc__gamma': ['scale', 0.001, 0.01, 0.1, 1]
}

# 5. Grid search with 5-fold cross-validation
grid = GridSearchCV(
    pipe,
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,     # use all CPU cores
    verbose=1
)

grid.fit(X_train, y_train)

# 6. Best settings
print("\n✅ Best parameters:", grid.best_params_)
print(f"✅ Best cross-val score: {grid.best_score_:.4f}")

# 7. Test on unseen data
y_pred = grid.predict(X_test)
print(f"\n✅ Test accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))