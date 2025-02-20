import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image in grayscale
img = cv2.imread("chair.jpg", cv2.IMREAD_GRAYSCALE)

plt.figure(figsize=(10,10))

def display_image(title, image, s, cmap='gray'):
    plt.subplot(3, 3, s)
    plt.imshow(image, cmap=cmap)
    plt.title(title)
    plt.axis("off")

display_image("Input Image", img, 1)

# Step 1: Apply Canny Edge Detection
edges = cv2.Canny(img, threshold1=50, threshold2=150)
display_image("Step 1: Canny Edge Detection", edges, 2)

# Step 2: Apply Morphological Closing to refine edges
kernel = np.ones((5,5), np.uint8)
edges_closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
display_image("Step 2: Morphological Closing", edges_closed, 3)

# Step 3: Find Contours
contours, _ = cv2.findContours(edges_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Create separate masks for each object
mask_chair = np.zeros_like(img)
mask_window = np.zeros_like(img)
mask_lamp = np.zeros_like(img)
mask_all = np.zeros_like(img)

contour1 = sorted(contours, key=cv2.contourArea, reverse=True)[:1]  # Chair
contour2 = sorted(contours, key=cv2.contourArea, reverse=True)[1:2]  # Window
contour3 = sorted(contours, key=cv2.contourArea, reverse=True)[2:3]  # Lamp
contour4 = sorted(contours, key=cv2.contourArea, reverse=True)[:3]  # All objects

cv2.drawContours(mask_chair, contour1, -1, (255, 255, 255), thickness=cv2.FILLED)
display_image("Step 3.1: Chair Contour", mask_chair, 4)

cv2.drawContours(mask_window, contour2, -1, (255, 255, 255), thickness=cv2.FILLED)
display_image("Step 3.2: Window Contour", mask_window, 5)

cv2.drawContours(mask_lamp, contour3, -1, (255, 255, 255), thickness=cv2.FILLED)
display_image("Step 3.3: Lamp Contour", mask_lamp, 6)

cv2.drawContours(mask_all, contour4, -1, (255, 255, 255), thickness=cv2.FILLED)
display_image("Step 3.4: All Objects Contour", mask_all, 7)

# Step 4: Extract the Chair using the Mask
extracted_chair = cv2.bitwise_and(img, img, mask=mask_all)
display_image("Step 4: Extracted Objects", extracted_chair, 8)

# Step 5: Apply Manual Thresholding
manual_threshold_value = 170  # Adjust as needed
_, manual_thresh = cv2.threshold(extracted_chair, manual_threshold_value, 255, cv2.THRESH_BINARY)
display_image("Step 5: Manual Thresholding", manual_thresh, 9)

plt.show()
