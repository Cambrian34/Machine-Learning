import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Dense, Reshape, Flatten, LeakyReLU, Input, BatchNormalization, Dropout, Conv2D
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam
import os

# --- Hyperparameters ---
img_rows = 128
img_cols = 128
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

X_train_resized = tf.image.resize(X_train, (128, 128))  # Resize to (128, 128, 1)


print(f"Training data shape: {X_train.shape}")

#from_logits=True


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LeakyReLU, BatchNormalization, Reshape, Conv2DTranspose

def build_generator(latent_dim, img_shape):
    model = Sequential(name="Generator")

    # Initial Dense layer to map latent space to a larger shape
    model.add(Dense(128 * 8 * 8, input_dim=latent_dim))  # Start with a larger spatial size (8x8)
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Reshape((8, 8, 128)))  # Reshape into a 3D tensor (height, width, channels)

    # First deconvolution (upsample)
    model.add(Conv2DTranspose(128, kernel_size=4, strides=2, padding='same'))  # 16x16
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))

    # Second deconvolution (upsample)
    model.add(Conv2DTranspose(64, kernel_size=4, strides=2, padding='same'))  # 32x32
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))

    # Third deconvolution (upsample)
    model.add(Conv2DTranspose(32, kernel_size=4, strides=2, padding='same'))  # 64x64
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))

    # Final output layer to match the image shape
    model.add(Conv2DTranspose(img_shape[2], kernel_size=4, strides=2, padding='same', activation='tanh'))  # 128x128 or 256x256

    model.summary()
    return model

"""
# --- Build the Generator ---
def build_generator(latent_dim, img_shape):
    model = Sequential(name="Generator")
    model.add(Dense(256, input_dim=latent_dim))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(1024))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(np.prod(img_shape), activation='tanh'))
    model.add(Reshape(img_shape))
    model.summary()
    return model

# --- Build the Discriminator ---
def build_discriminator(img_shape):
    model = Sequential(name="CNN_Discriminator")

    model.add(Conv2D(64, kernel_size=3, strides=2, input_shape=img_shape, padding="same"))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.3))

    model.add(Conv2D(128, kernel_size=3, strides=2, padding="same"))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.3))

    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))

    #model.add(Dense(1))  # Output: Probability (Real/Fake)

    model.summary()
    return model

"""
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, LeakyReLU, Dropout, Flatten, Dense, BatchNormalization

def build_discriminator(img_shape):
    model = Sequential(name="PatchGAN_Discriminator")

    # First layer: Conv2D with LeakyReLU and BatchNormalization for better convergence
    model.add(Conv2D(64, kernel_size=4, strides=2, input_shape=img_shape, padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.3))
    
    # Second layer
    model.add(Conv2D(128, kernel_size=4, strides=2, padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.3))
    
    # Third layer
    model.add(Conv2D(256, kernel_size=4, strides=2, padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.3))
    
    # Fourth layer
    model.add(Conv2D(512, kernel_size=4, strides=2, padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.3))

    # Flatten and add output layer
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))  # Output: Probability (Real/Fake)

    model.summary()
    return model

# --- Build and Compile Models ---
optimizer_D = Adam(learning_rate=learning_rate, beta_1=beta_1)
optimizer_G = Adam(learning_rate=learning_rate, beta_1=beta_1)

discriminator = build_discriminator(img_shape)
discriminator.compile(loss='binary_crossentropy', optimizer=optimizer_D, metrics=['accuracy'])

generator = build_generator(latent_dim, img_shape)

# Create the combined GAN model (Generator -> Discriminator)
discriminator.trainable = False  # Freeze discriminator weights

z = Input(shape=(latent_dim,))
img = generator(z)
validity = discriminator(img)

combined = Model(z, validity, name="Combined_GAN")
combined.compile(loss='binary_crossentropy', optimizer=optimizer_G)

print("\n--- Combined Model (Generator Training) ---")
combined.summary()

# --- @tf.function Optimization ---
@tf.function(reduce_retracing=True)
def train_discriminator(real_imgs, fake_imgs, valid, fake):
    with tf.GradientTape() as tape:
        real_output = discriminator(real_imgs)
        real_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)(valid, real_output)

        fake_output = discriminator(fake_imgs)
        fake_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)(fake, fake_output)

        d_loss = (real_loss + fake_loss) / 2

    # Check if the discriminator has trainable variables
    if discriminator.trainable_variables:
        discriminator_gradients = tape.gradient(d_loss, discriminator.trainable_variables)
        optimizer_D.apply_gradients(zip(discriminator_gradients, discriminator.trainable_variables))
    else:
        pass  # No action needed if there are no trainable variables

    real_accuracy = tf.reduce_mean(tf.cast(tf.greater_equal(real_output, 0), tf.float32))
    fake_accuracy = tf.reduce_mean(tf.cast(tf.less(fake_output, 0), tf.float32))
    d_accuracy = (real_accuracy + fake_accuracy) / 2

    return d_loss, d_accuracy

@tf.function(reduce_retracing=True)
def train_generator(noise, valid):
    with tf.GradientTape() as tape:
        fake_imgs = generator(noise, training=True)
        fake_output = discriminator(fake_imgs, training=True)
        g_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)(valid, fake_output)
        
    # Apply gradients
    generator_gradients = tape.gradient(g_loss, generator.trainable_variables)
    optimizer_G.apply_gradients(zip(generator_gradients, generator.trainable_variables))
    
    return g_loss
# --- Training Loop ---
valid = tf.ones((batch_size, 1)) * 0.9  # Real labels are 0.9 instead of 1.0
fake = tf.zeros((batch_size, 1))   # Labels for fake images

for epoch in range(epochs + 1):
    idx = np.random.randint(0, X_train.shape[0], batch_size)
    real_imgs = X_train[idx]
    real_imgs += np.random.normal(0, 0.1, real_imgs.shape)

    noise = np.random.normal(0, 1, (batch_size, latent_dim))
    fake_imgs = generator(noise, training=False)

    d_loss, d_acc = train_discriminator(real_imgs, fake_imgs, valid, fake)

    noise = np.random.normal(0, 1, (batch_size, latent_dim))
    g_loss = train_generator(noise, valid)

    if epoch % 100 == 0:
        print(f"{epoch} [D loss: {d_loss:.4f}, acc.: {100 * d_acc:.2f}%] [G loss: {g_loss:.4f}]")

    discriminator.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                          optimizer=optimizer_D,
                          metrics=['accuracy'])
    combined.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                     optimizer=optimizer_G)

    if epoch % sample_interval == 0:
        print(f"--- Saving sample images at epoch {epoch} ---")
        r, c = 5, 5
        noise = np.random.normal(0, 1, (r * c, latent_dim))
        gen_imgs = generator(noise, training=False)

        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(r, c, figsize=(r, c))
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt, :, :, 0], cmap='gray')
                axs[i,j].axis('off')
                cnt += 1
        fig.suptitle(f"GAN: Generated MNIST digits at Epoch {epoch}")
        save_path = os.path.join(save_dir, f"mnist_{epoch:06d}.png")
        fig.savefig(save_path)
        print(f"Saved images to {save_path}")
        plt.close(fig)

print("--- Training Finished ---")