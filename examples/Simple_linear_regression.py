import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Sample dataset: Heights (cm) and corresponding Weights (kg)
heights = np.array([150, 160, 170, 180, 190]).reshape(-1, 1)  # Feature (Input)
weights = np.array([45, 55, 60, 70, 75])  # Target (Output)

# Create and train the model
model = LinearRegression()
model.fit(heights, weights)

# Predict weight for a new height (e.g., 175 cm)
predicted_weight = model.predict([[175]])
print(f"Predicted weight for 175 cm: {predicted_weight[0]:.2f} kg")

# Plot the data and regression line
plt.scatter(heights, weights, color='blue', label='Actual data')
plt.plot(heights, model.predict(heights), color='red', label='Regression Line')
plt.xlabel('Height (cm)')
plt.ylabel('Weight (kg)')
plt.legend()
plt.show()