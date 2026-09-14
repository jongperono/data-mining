import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier


### -------------------------- ###

# Example 2: Feature Importance (The Killer Feature)

### -------------------------- ###

# 1. Load data
iris = load_iris()
X, y = iris.data, iris.target
feature_names = iris.feature_names

# 2. Train
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X, y)

# 3. Get feature importances
importances = rf.feature_importances_

# 4. Show as sorted table
df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values('Importance', ascending=False)

print(df.to_string(index=False))

# 5. Plot
plt.figure(figsize=(8, 5))
plt.barh(df['Feature'], df['Importance'], color='forestgreen')
plt.xlabel('Importance')
plt.title('Feature Importance — Random Forest')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()