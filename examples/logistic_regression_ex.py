import numpy as np
from sklearn.linear_model import LogisticRegression

# Sample dataset: Number of "free" words in emails vs. Spam or Not Spam
X = np.array([[1], [3], [5], [7], [9], [11]])  # Feature: "free" word count
y = np.array([0, 0, 0, 1, 1, 1])  # Labels: 0 (Not Spam), 1 (Spam)

# Create and train the logistic regression model
model = LogisticRegression()
model.fit(X, y)

# Predict probability of spam for an email with 6 "free" words
probability = model.predict_proba([[6]])[0][1]
print(f"Probability that an email with 6 'free' words is spam: {probability:.2f}")

# Predict class (0 or 1)
prediction = model.predict([[4]])
print(f"Predicted class: {'Spam' if prediction[0] == 1 else 'Not Spam'}")