"""A Random Forest is a collection of multiple Decision Trees, each trained on different parts of the data. 
It makes decisions based on the majority vote of these trees, reducing the risk of overfitting."""

from sklearn.ensemble import RandomForestClassifier
import numpy as np


# Features: [Age, Budget ($)]
X = np.array([
    [18, 200], [21, 500], [22, 150], [16, 100], [19, 120], [25, 300] , [2,1000] , [75, 1000]
])

# Labels: 1 = Buys longboard, 0 = Doesn't buy
y = np.array([1, 1, 1, 0, 0, 1, 0,0])


# Train Random Forest (with 10 trees)
model = RandomForestClassifier(n_estimators=10, random_state=0)
model.fit(X, y)

# Predict for a new person (Age: 20, Budget: $160)
prediction = model.predict([[20, 160]])
print("Will they buy a longboard?", "Yes" if prediction[0] == 1 else "No")