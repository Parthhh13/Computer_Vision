import numpy as np
import matplotlib.pyplot as plt
import cv2

# Load and denoise the image
image = cv2.imread("uneven.jpg", cv2.IMREAD_GRAYSCALE)
#image = cv2.resize(image, (256, 256))
image = cv2.GaussianBlur(image, (5, 5), 0)

def moving_average_thresholding(image, kernel_size):
    pad = kernel_size // 2
    padded_image = np.pad(image, pad, mode='reflect')
    output_image = np.zeros_like(image)
    pad_rows, pad_cols = padded_image.shape

    for row in range(pad, pad_rows - pad):
        for col in range(pad, pad_cols - pad):
            kernel = padded_image[row - pad:row + pad + 1, col - pad:col + pad + 1]
            
            # Calculate the moving average (mean of the kernel)
            mean = np.mean(kernel)

            # Apply thresholding based on the moving average
            output_image[row - pad, col - pad] = 255 if padded_image[row, col] > mean else 0

    return output_image

images = [image]
title = ["Original Image"]
kernel_size = 3  # Set kernel size for the moving average window
output = moving_average_thresholding(image, kernel_size)
images.append(output)
title.append(f"Kernel size = {kernel_size}")

# Plot the images
plt.figure(figsize=(8, 6))

for i in range(len(images)):
    plt.subplot(1, 2, i + 1)
    plt.title(title[i])
    plt.imshow(images[i], cmap='gray')
    plt.axis('off')
plt.tight_layout()
plt.show()
