import numpy as np
import cv2
import matplotlib.pyplot as plt
import math

image = cv2.imread("lenna.jpg",
                   cv2.IMREAD_GRAYSCALE)
image = cv2.resize(image, (300, 300)) 

def translation(image,tx,ty):
    translated_image = np.zeros_like(image)  # Create a new array for the translated image
    rows, cols = image.shape
    for row in range(rows):
        for col in range(cols):
            trans = np.array(
                [[1, 0, tx],
                 [0, 1, ty],
                 [0, 0, 1]] ) 
            initial=np.array(
                [[row],
                 [col],
                 [1]] )
            new=np.matmul(trans,initial)

            new_row, new_col = int(new[0][0]), int(new[1][0])
            if 0 <= new_row < rows and 0 <= new_col < cols:
                translated_image[new_row, new_col] = image[row, col] 
    
    return translated_image

def scaling(image,cx,cy):
    rows, cols = image.shape
    new_rows, new_cols = int(rows * cx), int(cols * cy)
    scaled_image = np.all((new_rows, new_cols),-1)
    for row in range(rows):
        for col in range(cols):
            trans = np.array(
                [[cx, 0, 0],
                 [0, cy, 0],
                 [0, 0, 1]] ) 
            initial=np.array(
                [[row],
                 [col],
                 [1]] )
            new=np.matmul(trans,initial)

            new_row, new_col = int(new[0][0]), int(new[1][0])
            if 0 <= new_row < new_rows and 0 <= new_col < new_cols:
                scaled_image[new_row, new_col] = image[row, col] 
    
    return scaled_image

def rotation(image,angle):
    theta = math.radians(angle)
    sin = math.sin(theta)
    cos = math.cos(theta)
    rotated_image = np.ones_like(image)  # Create a new array for the translated image
    rows, cols = image.shape
    for row in range(rows):
        for col in range(cols):
            trans = np.array(
                [[cos, sin, 0],
                 [-sin, cos, 0],
                 [0, 0, 1]] ) 
            initial=np.array(
                [[row],
                 [col],
                 [1]] )
            new=np.matmul(trans,initial)

            new_row, new_col = int(new[0][0]), int(new[1][0])
            if 0 <= new_row < rows and 0 <= new_col < cols:
                rotated_image[new_row, new_col] = image[row, col] 
    
    return rotated_image

def rotation2(image, angle):
    theta = math.radians(angle)
    sin = math.sin(theta)
    cos = math.cos(theta)
    
    rows, cols = image.shape
    
    # Calculate the new bounding box for the rotated image
    new_rows = int(abs(rows * cos) + abs(cols * sin))
    new_cols = int(abs(cols * cos) + abs(rows * sin))
    
    # Create an empty image for the rotated result (initialized with zeros)
    rotated_image = np.zeros((new_rows, new_cols), dtype=image.dtype)
    
    # Find the center of the original image
    center_x, center_y = cols // 2, rows // 2
    new_center_x, new_center_y = new_cols // 2, new_rows // 2
    
    for row in range(rows):
        for col in range(cols):
            # Translate coordinates so that the rotation happens around the center
            translated_row = row - center_y
            translated_col = col - center_x
            
            # Apply the rotation matrix
            rotated_row = translated_row * cos - translated_col * sin
            rotated_col = translated_row * sin + translated_col * cos
            
            # Translate back to the center of the new image
            new_row = int(rotated_row + new_center_y)
            new_col = int(rotated_col + new_center_x)
            
            # If the new pixel coordinates are within the bounds of the rotated image
            if 0 <= new_row < new_rows and 0 <= new_col < new_cols:
                rotated_image[new_row, new_col] = image[row, col]
    
    return rotated_image



'''cv2.imshow("Original", image)
cv2.imshow("Translated", translation(image,-20,30))
cv2.imshow("Scaled", scaling(image,2,3))
cv2.waitKey(0)
cv2.destroyAllWindows()'''

plt.figure(figsize=(6, 4))

# Original Image
plt.subplot(2, 3, 1)
plt.imshow(image, cmap='gray')
plt.title("Original")
plt.axis("off")

# Translated Image
plt.subplot(2, 3, 2)
plt.imshow(translation(image,-20,30), cmap='gray')
plt.title("Translated")
plt.axis("off")

# Scaled Image
plt.subplot(2, 3, 3)
plt.imshow(scaling(image,2,2), cmap='gray')
plt.title("Scaled")
plt.axis("off")

# Rotated Image
plt.subplot(2, 3, 4)
plt.imshow(rotation(image,30), cmap='gray')
plt.title("Basic Rotated")
plt.axis("off")

# Rotated Image
plt.subplot(2, 3, 5)
plt.imshow(rotation2(image,30), cmap='gray')
plt.title("Advance Rotated")
plt.axis("off")

# Show the plots
plt.tight_layout()
plt.show()

