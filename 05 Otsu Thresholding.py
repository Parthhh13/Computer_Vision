import numpy as np
import matplotlib.pyplot as plt
import cv2

# Load the image
<<<<<<< HEAD
image = cv2.imread(r"E:\Study Material\NIIT\6th Semester\Computer Vision\Lab\Crack.jpg", cv2.IMREAD_GRAYSCALE)
image=cv2.resize(image,(256,256))
=======
image = cv2.imread(r"chair.jpg", cv2.IMREAD_GRAYSCALE)
#image=cv2.resize(image,(256,256))
>>>>>>> 976db7b74fb8d528c8cef10d7b0d669469b56a50


def otsu_thresholding(image):
    rows,cols=image.shape

    #Finding Probability for each intensity
    prob=np.zeros(256)
    for i in range(256):
        prob[i]=np.sum(image==i)/(rows*cols)
    
    #Initialize with a threshold value k
    max_variance = 0
    optimal_threshold = 0

    for threshold in range(1,255):

        P1=np.sum(prob[0:threshold+1])

        culMean=np.sum(np.arange(threshold+1)*prob[0:threshold+1])

        globMean=np.sum(np.arange(256)*prob)

        if P1==1 or P1==0:
            continue
        inBWvar=(globMean*P1-culMean)**2/(P1*(1-P1))

        if inBWvar>max_variance:
            max_variance=inBWvar
            optimal_threshold=threshold

    binary_image = np.where(image >90, 255, 0)

    return optimal_threshold,binary_image



plt.figure(figsize=(8, 6))

# Original Image
plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(image, cmap='gray')
plt.axis('off')

#Otsu Thresholding
threshold,binary_image=otsu_thresholding(image)
plt.subplot(1, 2, 2)
plt.title(f"Threshold value={threshold}")
plt.imshow(binary_image, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()