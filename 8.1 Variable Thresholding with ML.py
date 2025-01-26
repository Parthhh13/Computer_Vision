import numpy as np
import matplotlib.pyplot as plt
import cv2
from sklearn.metrics import mean_squared_error


# Load and denoise the image
image = cv2.imread("lenna.jpg", cv2.IMREAD_GRAYSCALE)
image = cv2.resize(image, (256, 256))
# image = cv2.GaussianBlur(image, (5, 5), 0)

# Example ground truth image (you should replace this with a real ground truth if you have one)
# For now, I'll use a simple binarized version of the image as a "ground truth."
ground_truth = np.where(image > 69, 255, 0)  # Example thresholding for ground truth

def variable_thresholding(image, kernel_size, a, b):
    pad = kernel_size // 2
    padded_image = np.pad(image, pad, mode='reflect')
    output_image = np.zeros_like(image)
    pad_rows, pad_cols = padded_image.shape

    for row in range(pad, pad_rows - pad):
        for col in range(pad, pad_cols - pad):
            kernel = padded_image[row - pad:row + pad + 1, col - pad:col + pad + 1]

            mean = np.mean(kernel)
            std = np.std(kernel)

            threshold = a * std + b * mean

            output_image[row - pad][col - pad] = np.where(padded_image[row][col] > threshold, 255, 0)

    return output_image


def optimize_ab(image, ground_truth):
    best_a = 0
    best_b = 0
    best_score = float('inf')  # Initial high MSE to be minimized

    # Define the search space for a and b
    a_values = np.linspace(0.1, 1.0, 10)  # 10 values for a between 0.1 and 1.0
    b_values = np.linspace(0.1, 1.0, 10)  # 10 values for b between 0.1 and 1.0

    # Loop over all combinations of a and b
    for a in a_values:
        for b in b_values:
            output = variable_thresholding(image, kernel_size=15, a=a, b=b)

            # Compute MSE between the thresholded image and the ground truth
            score = mean_squared_error(ground_truth.flatten(), output.flatten())

            # Update best parameters if we find a better score
            if score < best_score:
                best_score = score
                best_a, best_b = a, b

    return best_a, best_b


# Optimize a and b values
best_a, best_b = optimize_ab(image, ground_truth)
print(f"Optimal a, b: {best_a}, {best_b}")

# Generate the output using the optimized values of a and b
output_optimized = variable_thresholding(image, kernel_size=15, a=best_a, b=best_b)

# Display the original and thresholded images
images = [image, output_optimized]
title = ["Original Image", f"Thresholded Image (a={best_a:.2f}, b={best_b:.2f})"]

plt.figure(figsize=(8, 6))

for i in range(len(images)):
    plt.subplot(1, 2, i + 1)
    plt.title(title[i])
    plt.imshow(images[i], cmap='gray')
    plt.axis('off')
plt.tight_layout()
plt.show()
