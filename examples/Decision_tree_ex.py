from sklearn.tree import DecisionTreeClassifier
import numpy as np

# Features: [Age, Budget ($)]
X = np.array([
    [18, 200], [21, 500], [22, 150], [16, 100], [19, 120], [25, 300] , [2,1000] , [75, 1000]
])

# Labels: 1 = Buys longboard, 0 = Doesn't buy
y = np.array([1, 1, 1, 0, 0, 1, 0,0])

# Train Decision Tree
model = DecisionTreeClassifier()
model.fit(X, y)

# Predict for a new person (Age: 20, Budget: $160)
prediction = model.predict([[700, 10]])
print("Will they buy a longboard?", "Yes" if prediction[0] == 1 else "No")