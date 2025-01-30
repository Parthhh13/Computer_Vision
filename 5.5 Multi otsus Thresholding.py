import numpy as np
import matplotlib.pyplot as plt
import cv2
from itertools import combinations

# Load the image
image = cv2.imread("trees.jpg", cv2.IMREAD_GRAYSCALE)
image = cv2.resize(image, (256, 256))

def otsu_thresholding(image, num_classes):
    rows, cols = image.shape

    # Finding probability for each intensity
    prob = np.zeros(256)
    for i in range(256):
        prob[i] = np.sum(image == i) / (rows * cols)

    # Initialize with a threshold value k
    max_variance = 0
    optimal_thresholds = []

    # Generate all combinations of thresholds for n classes
    threshold_combinations = combinations(range(1, 255), num_classes - 1)

    for thresholds in threshold_combinations:
        thresholds = [0] + list(thresholds) + [255]  # Add start and end points

        P = []
        culMean = []
        for i in range(len(thresholds) - 1):
            start, end = thresholds[i], thresholds[i + 1]
            P.append(np.sum(prob[start:end]))
            culMean.append(np.sum(np.arange(start, end) * prob[start:end]))

        globMean = np.sum(np.arange(256) * prob)

        within_class_variance = 0
        for i in range(len(P)):
            if P[i] > 0:
                within_class_variance += (globMean * P[i] - culMean[i]) ** 2 / P[i]

        if within_class_variance > max_variance:
            max_variance = within_class_variance
            optimal_thresholds = thresholds[1:-1]  # Exclude 0 and 255

    # Segment the image using the optimal thresholds
    segmented_image = np.zeros_like(image)
    for i, threshold in enumerate(optimal_thresholds):
        if i == 0:
            segmented_image[image <= threshold] = i * (255 // num_classes)
        else:
            segmented_image[(image > optimal_thresholds[i - 1]) & (image <= threshold)] = i * (255 // num_classes)

    segmented_image[image > optimal_thresholds[-1]] = (num_classes - 1) * (255 // num_classes)

    return optimal_thresholds, segmented_image

# Number of classes for segmentation
num_classes = 3

plt.figure(figsize=(12, 8))

# Original Image
plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(image, cmap='gray')
plt.axis('off')

# Otsu Thresholding for n classes
thresholds, segmented_image = otsu_thresholding(image, num_classes)
plt.subplot(1, 2, 2)
plt.title(f"Thresholds: {thresholds}")
plt.imshow(segmented_image, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()
