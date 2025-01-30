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