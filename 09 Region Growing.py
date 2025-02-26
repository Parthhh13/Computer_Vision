<<<<<<< HEAD
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
=======
import cv2
import numpy as np
import matplotlib.pyplot as plt

def region_growing(image, seed_points, lower_threshold, upper_threshold):
    # Get the image dimensions
    height, width = image.shape
   
    # Create an output image initialized to 0
    output = np.zeros_like(image)
   
    # Define 8-connected neighborhood
    neighbors = [(-1, -1), (-1, 0), (-1, 1),
                 (0, -1),          (0, 1),
                 (1, -1),  (1, 0), (1, 1)]
   
    # Create a binary mask where pixels satisfy the predicate
    predicate_mask = (image >= lower_threshold) & (image <= upper_threshold)
   
    # Label each seed point
    for seed in seed_points:
        y, x = seed
        if output[y, x] == 0:  # If not already labeled
            output[y, x] = 255  # Mark seed points with 255
   
    # Create a copy of the initial seed points for visualization
    initial_seed_image = output.copy()
   
    # Create an image showing seed points after applying predicate
    seed_after_predicate = np.zeros_like(image)
    for y, x in seed_points:
        if predicate_mask[y, x]:
            seed_after_predicate[y, x] = 255
   
    # Iterate until no more changes occur
    changed = True
    while changed:
        changed = False
        for y in range(height):
            for x in range(width):
                if output[y, x] == 255:
                    for dy, dx in neighbors:
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < height and 0 <= nx < width:
                            if output[ny, nx] == 0 and predicate_mask[ny, nx]:
                                output[ny, nx] = 255
                                changed = True
   
    return output, initial_seed_image, seed_after_predicate

# Load an image
image = cv2.imread("Crack.jpg", cv2.IMREAD_GRAYSCALE)

# Check if the image was loaded successfully

    # Define seed points (you can manually select or automatically detect them)
seed_points = [(50, 50), (100, 100)]  # Example seed points (y, x)

    # Define thresholds
lower_threshold = 100
upper_threshold = 200

    # Apply region growing
segmented_image, initial_seed_image, seed_after_predicate = region_growing(image, seed_points, lower_threshold, upper_threshold)

    # Display the results
plt.figure(figsize=(20, 5))

    # Original image
plt.subplot(1, 4, 1)
plt.title('Original Image')
plt.imshow(image, cmap='gray')

    # Initial seed points
plt.subplot(1, 4, 2)
plt.title('Initial Seed Points')
plt.imshow(initial_seed_image, cmap='gray')

    # Seed points after predicate
plt.subplot(1, 4, 3)
plt.title('Seeds After Predicate')
plt.imshow(seed_after_predicate, cmap='gray')

    # Final segmented image
plt.subplot(1, 4, 4)
plt.title('Segmented Image')
plt.imshow(segmented_image, cmap='gray')

plt.show()
>>>>>>> 976db7b74fb8d528c8cef10d7b0d669469b56a50
