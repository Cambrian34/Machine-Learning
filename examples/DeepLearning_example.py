import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np

# Load the MNIST dataset (handwritten digits 0-9)
(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

# Normalize data (scale pixel values to 0-1 range)
X_train, X_test = X_train / 255.0, X_test / 255.0

# Build the neural network model
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),  # Input layer (flatten 28x28 images)
    keras.layers.Dense(128, activation='relu'),  # Hidden layer
    keras.layers.Dense(10, activation='softmax')  # Output layer (10 classes)
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=5)

# Evaluate on test data
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_acc:.2f}")

# Show a sample image
plt.imshow(X_test[0], cmap='gray')
plt.title(f"Predicted: {np.argmax(model.predict(X_test[0].reshape(1,28,28)))}")
plt.show()