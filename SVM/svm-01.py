from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report


### -------------------------- ###

# Example 1: Basic SVM (Iris Dataset)

### -------------------------- ###


# 1. Load data
iris = datasets.load_iris()
print(iris.data)
print('----------')
print(iris.target)
X = iris.data       # 4 features
y = iris.target     # 3 classes

# 2. Split into train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# 3. Create SVM model
model = SVC(kernel='rbf', C=1.0, gamma='scale')

# 4. Train
model.fit(X_train, y_train)

# 5. Predict
y_pred = model.predict(X_test)

# 6. Evaluate
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))