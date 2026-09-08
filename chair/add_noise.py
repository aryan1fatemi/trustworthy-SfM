import cv2
import numpy as np
import matplotlib.pyplot as plt

def add_gaussian_noise(image, sigma=25.0):
    """
    Adds Gaussian additive measurement noise: I_noisy = I_true + eta, where eta ~ N(0, sigma^2)
    """
    # Convert image to float32 to prevent unsigned 8-bit integer underflow/overflow
    img_float = image.astype(np.float32)
    
    # Generate Gaussian noise array matching image dimensions
    noise = np.random.normal(loc=0.0, scale=sigma, size=image.shape)
    
    # Add noise to image
    noisy_img = img_float + noise
    
    # Clip pixel values back to valid range [0, 255] and convert to uint8
    noisy_img_clipped = np.clip(noisy_img, 0, 255).astype(np.uint8)
    return noisy_img_clipped, noise

def add_interval_noise(image, epsilon=30.0):
    """
    Adds Bounded Uniform Noise: I_noisy = I_true + eta, where eta in [-epsilon, epsilon]
    """
    img_float = image.astype(np.float32)
    
    # Generate uniform noise bounded in [-epsilon, +epsilon]
    noise = np.random.uniform(low=-epsilon, high=epsilon, size=image.shape)
    
    noisy_img = img_float + noise
    noisy_img_clipped = np.clip(noisy_img, 0, 255).astype(np.uint8)
    return noisy_img_clipped, noise

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# 1. Load the actual image from images/r_82.png
# -----------------------------------------------------------------------------
image_path = os.path.join("images", "r_82.png")

# Load in grayscale (0 flag in cv2.imread loads image directly as 2D intensity grid)
clean_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Check if image was successfully loaded
if clean_image is None:
    raise FileNotFoundError(f"Could not find or open the image at '{image_path}'. "
                            f"Please verify the directory structure and file name.")

# -----------------------------------------------------------------------------
# 2. Apply Noise Transformations
# -----------------------------------------------------------------------------
# Apply Gaussian Noise (sigma = 20 pixels)
noisy_gaussian, g_noise = add_gaussian_noise(clean_image, sigma=20.0)

# Apply Bounded Interval Noise (epsilon = 30 pixels)
noisy_interval, i_noise = add_interval_noise(clean_image, epsilon=30.0)

# -----------------------------------------------------------------------------
# 3. Plot Results Side-by-Side
# -----------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

axes[0].imshow(clean_image, cmap='gray', vmin=0, vmax=255)
axes[0].set_title("Clean Image ($I^{\\text{true}}$)\n`r_82.png`")
axes[0].axis('off')

#axes[1].imshow(noisy_gaussian, cmap='gray', vmin=0, vmax=255)
#axes[1].set_title("Gaussian Noise\n$\\eta \\sim \\mathcal{N}(0, 20^2)$")
#axes[1].axis('off')

axes[1].imshow(noisy_interval, cmap='gray', vmin=0, vmax=255)
axes[1].set_title("Interval Noise\n$\\eta \\in [-30, +30]$")
axes[1].axis('off')

plt.tight_layout()
plt.show()