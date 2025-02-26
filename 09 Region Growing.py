import numpy as np
import cv2
from scipy.ndimage import label

def region_growing(image, seed_points, predicate):
    # Initialize the seed image S
    S = np.zeros_like(image, dtype=np.uint8)
    for seed in seed_points:
        S[seed] = 1

    # Create the predicate image fQ
    fQ = np.zeros_like(image, dtype=np.uint8)
    fQ[predicate(image)] = 1

    # Initialize the output image g
    g = np.zeros_like(image, dtype=np.uint8)

    # For each seed point, perform region growing
    for seed in seed_points:
        region_mask = np.zeros_like(image, dtype=np.uint8)
        region_mask[seed] = 1

        while True:
            boundary = cv2.dilate(region_mask, np.ones((3, 3), np.uint8)) - region_mask
            new_pixels = np.logical_and(boundary, fQ)
            if not np.any(new_pixels):
                break
            region_mask[new_pixels] = 1

        g[region_mask == 1] = 1

    # Label each connected component in g
    labeled_image, num_features = label(g, structure=np.ones((3, 3)))

    return labeled_image

# Load the image
image = cv2.imread('Crack.jpg', cv2.IMREAD_GRAYSCALE)
if image is None:
    raise ValueError("Image not loaded. Check the file path.")

# Step 1: Threshold the image
_, thresholded = cv2.threshold(image, 200, 255, cv2.THRESH_BINARY)

# Step 2: Erode the thresholded image to get single-pixel seeds
kernel = np.ones((3, 3), np.uint8)
eroded = cv2.erode(thresholded, kernel, iterations=1)

# Step 3: Extract seed points
seed_points = list(zip(*np.where(eroded > 0)))

# Step 4: Define a predicate for region growing
predicate = lambda img: np.abs(img - image) <= 2

# Step 5: Apply region growing
segmented_image = region_growing(image, seed_points, predicate)

# Display the results
cv2.imshow('Thresholded Image', thresholded)
cv2.imshow('Eroded Image', eroded)
cv2.imshow('Segmented Image', segmented_image)
cv2.waitKey(0)
cv2.destroyAllWindows()