import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Dense, Reshape, Flatten, LeakyReLU, Input, BatchNormalization,Dropout
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam
import os

# --- Hyperparameters ---
img_rows = 28
img_cols = 28
channels = 1
img_shape = (img_rows, img_cols, channels)
latent_dim = 100          # Dimension of the noise vector
learning_rate = 0.0002
beta_1 = 0.5             # Beta1 for Adam optimizer (common in GANs)
epochs = 20000           # Reduced for quicker demonstration, increase for better results (e.g., 50000+)
batch_size = 128         # Common batch size
sample_interval = 1000   # How often to save generated images
save_dir = 'mnist_gan_images' # Directory to save images

# Create save directory if it doesn't exist
os.makedirs(save_dir, exist_ok=True)

# --- Load and Preprocess MNIST Data ---
(X_train, _), (_, _) = mnist.load_data()

# Normalize pixel values to be between -1 and 1
X_train = (X_train.astype(np.float32) - 127.5) / 127.5
X_train = np.expand_dims(X_train, axis=-1)  # Add channel dimension: (60000, 28, 28, 1)

print(f"Training data shape: {X_train.shape}")

# --- Build the Generator ---
def build_generator(latent_dim, img_shape):
    model = Sequential(name="Generator")
    # Use Dense layers to generate pixel data from noise
    model.add(Dense(256, input_dim=latent_dim))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8)) # Added BatchNorm
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8)) # Added BatchNorm
    model.add(Dense(1024))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8)) # Added BatchNorm
    # Output layer: number of neurons = img_rows * img_cols * channels
    model.add(Dense(np.prod(img_shape), activation='tanh'))
    # Reshape output to image dimensions
    model.add(Reshape(img_shape))
    model.summary() # Print model summary
    return model

# --- Build the Discriminator ---
from tensorflow.keras.layers import Conv2D # Make sure Conv2D is imported

def build_discriminator(img_shape):
    model = Sequential(name="CNN_Discriminator")

    # Input: 28x28x1
    model.add(Conv2D(64, kernel_size=3, strides=2, input_shape=img_shape, padding="same")) # Output: 14x14x64
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.3))

    model.add(Conv2D(128, kernel_size=3, strides=2, padding="same")) # Output: 7x7x128
    # No BatchNorm usually needed if using Dropout and LeakyReLU
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.3))

    # Flatten the feature map
    model.add(Flatten()) # Output: 7*7*128 = 6272

    # Dense layer for classification
    model.add(Dense(1, activation='sigmoid')) # Output: Probability (Real/Fake)

    model.summary()
    return model
# --- Build and Compile Models ---

# Optimizers (using separate instances is good practice)
optimizer_D = Adam(learning_rate=learning_rate, beta_1=beta_1)
optimizer_G = Adam(learning_rate=learning_rate, beta_1=beta_1)

# Build and compile the Discriminator
discriminator = build_discriminator(img_shape)
discriminator.compile(loss='binary_crossentropy',
                      optimizer=optimizer_D,
                      metrics=['accuracy'])

# Build the Generator
generator = build_generator(latent_dim, img_shape)

# Create the combined GAN model (Generator -> Discriminator)
# For the combined model, we only train the generator.
discriminator.trainable = False # Freeze discriminator weights

# Input noise vector z
z = Input(shape=(latent_dim,))
# Generated image based on noise
img = generator(z)
# Discriminator's prediction on the generated image
validity = discriminator(img)

# Combined model (stacks generator and discriminator)
# Trains generator to fool discriminator
combined = Model(z, validity, name="Combined_GAN")
combined.compile(loss='binary_crossentropy', optimizer=optimizer_G)

print("\n--- Combined Model (Generator Training) ---")
combined.summary()

# --- Training Loop ---

# Adversarial ground truths
valid = tf.ones((batch_size, 1)) * 0.9  # Real labels are 0.9 instead of 1.0
fake = tf.zeros((batch_size, 1))   # Labels for fake images

for epoch in range(epochs + 1):

    # --- Train Discriminator ---

    # Select a random batch of real images
    idx = np.random.randint(0, X_train.shape[0], batch_size)
    real_imgs = X_train[idx]

    # Generate a batch of new fake images
    noise = np.random.normal(0, 1, (batch_size, latent_dim))
    # Use training=False to ensure BatchNorm layers use moving averages during generation if present
    fake_imgs = generator(noise, training=False)

    # Train the discriminator on real images (label as 1)
    d_loss_real = discriminator.train_on_batch(real_imgs, valid)
    # Train the discriminator on fake images (label as 0)
    d_loss_fake = discriminator.train_on_batch(fake_imgs, fake)
    # Average discriminator loss
    d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

    # --- Train Generator ---

    # Generate a new batch of noise
    noise = np.random.normal(0, 1, (batch_size, latent_dim))

    # Train the generator (via combined model) to make discriminator classify fake images as real (label as 1)
    g_loss = combined.train_on_batch(noise, valid)

    # --- Progress ---
    if epoch % 100 == 0: # Print more often initially
        print(f"{epoch} [D loss: {d_loss[0]:.4f}, acc.: {100 * d_loss[1]:.2f}%] [G loss: {g_loss:.4f}]")

    # --- Save Generated Images Sample ---
    if epoch % sample_interval == 0:
        print(f"--- Saving sample images at epoch {epoch} ---")
        # Generate a grid of images
        r, c = 5, 5 # Grid size
        noise = np.random.normal(0, 1, (r * c, latent_dim))
        gen_imgs = generator(noise, training=False) # Use training=False

        # Rescale images from [-1, 1] to [0, 1] for plotting
        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(r, c, figsize=(r, c))
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt, :, :, 0], cmap='gray')
                axs[i,j].axis('off')
                cnt += 1
        fig.suptitle(f"GAN: Generated MNIST digits at Epoch {epoch}")
        # Save the figure
        save_path = os.path.join(save_dir, f"mnist_{epoch:06d}.png")
        fig.savefig(save_path)
        print(f"Saved images to {save_path}")
        # plt.show() # Optionally display the plot
        plt.close(fig) # Close the figure to free memory

print("--- Training Finished ---")