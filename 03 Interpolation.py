import numpy as np
import matplotlib.pyplot as plt
import cv2

# Load the image
image = cv2.imread("lenna.jpg", cv2.IMREAD_GRAYSCALE)
image=cv2.resize(image,(256,256))


# Backward warping function
def backward_warping(image, cx, cy, method='nn'):
    rows, cols = image.shape
    new_rows, new_cols = int(rows * cx), int(cols * cy)
    scaled_image = np.zeros((new_rows, new_cols), dtype=image.dtype)

    for new_row in range(new_rows):
        for new_col in range(new_cols):
            # Map the target pixel in scaled image to the source image
            inv_trans = np.array([[1 / cx, 0, 0],
                                  [0, 1 / cy, 0],
                                  [0, 0, 1]])
            target = np.array([[new_row], [new_col], [1]])
            source_coords = np.matmul(inv_trans, target)
            src_row, src_col = source_coords[0][0], source_coords[1][0]

            # Apply the selected interpolation method
            if method == 'nn':  # Nearest Neighbor
                scaled_image[new_row, new_col] = nn_interpolation(image, src_row, src_col)
            elif method == 'bilinear':  # Bilinear Interpolation
                scaled_image[new_row, new_col] = bilinear_interpolation(image, src_row, src_col)

    return scaled_image


# Nearest Neighbor Interpolation
def nn_interpolation(image, x, y):
    # Find the nearest pixel in the source image
    nearest_row, nearest_col = round(x), round(y)
    rows, cols = image.shape
    # Boundary check
    if 0 <= nearest_row < rows and 0 <= nearest_col < cols:
        return image[nearest_row, nearest_col]
    return 0


# Bilinear Interpolation
def bilinear_interpolation(image, x, y):
    rows, cols = image.shape
    x0, y0 = int(np.floor(x)), int(np.floor(y))
    x1, y1 = x0 + 1, y0 + 1

    # Boundary check
    if x0 < 0 or y0 < 0 or x1 >= rows or y1 >= cols:
        return 0

    # Calculate distances
    dx = x - x0
    dy = y - y0

    # Intensities of the four neighbors
    I00 = image[x0, y0] if 0 <= x0 < rows and 0 <= y0 < cols else 0
    I01 = image[x0, y1] if 0 <= x0 < rows and 0 <= y1 < cols else 0
    I10 = image[x1, y0] if 0 <= x1 < rows and 0 <= y0 < cols else 0
    I11 = image[x1, y1] if 0 <= x1 < rows and 0 <= y1 < cols else 0

    # Bilinear interpolation formula
    interpolated_value = (
        (1 - dx) * (1 - dy) * I00 +
        dx * (1 - dy) * I10 +
        (1 - dx) * dy * I01 +
        dx * dy * I11
    )
    return interpolated_value


# Display results
plt.figure(figsize=(8, 6))

# Original Image
plt.subplot(1, 3, 1)
plt.title("Original Image")
plt.imshow(image, cmap='gray')
plt.axis('off')

# Scaled Image using Nearest Neighbor
scaled_nn = backward_warping(image, 2, 2, method='nn')
plt.subplot(1, 3, 2)
plt.title("NN Interpolation")
plt.imshow(scaled_nn, cmap='gray')
plt.axis('off')

# Scaled Image using Bilinear Interpolation
scaled_bilinear = backward_warping(image, 2, 2, method='bilinear')
plt.subplot(1, 3, 3)
plt.title("Bilinear Interpolation")
plt.imshow(scaled_bilinear, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()
