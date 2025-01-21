import numpy as np
import matplotlib.pyplot as plt
import cv2

# Load and denoise the image
image = cv2.imread("sample.jpg", cv2.IMREAD_GRAYSCALE)
image=cv2.resize(image,(256,256))
image=255-image

blur_image=cv2.GaussianBlur(image,(5,5),0)

#Find the binary edge matrix
grad_X=cv2.Sobel(blur_image,cv2.CV_64F,1,0,ksize=3)
grad_y=cv2.Sobel(blur_image,cv2.CV_64F,0,1,ksize=3)

magnitude=(grad_X**2+grad_y**2)**0.5
threshold_mag=0.95*np.max(magnitude)

bin_edges=np.where(magnitude>threshold_mag,1,0)
bin_edges_display=np.where(magnitude>threshold_mag,255,0)

#Mask with the orignal image
edges=bin_edges*image

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

    binary_image = np.where(image > optimal_threshold, 255, 0)

    return optimal_threshold,binary_image

threshold,_=otsu_thresholding(edges)

final=np.where(image>threshold,255,0)

plt.figure(figsize=(8, 6))

# Original Image
plt.subplot(2, 2, 1)
plt.title("Original Image")
plt.imshow(image, cmap='gray')
plt.axis('off')

#Otsu Thresholding
plt.subplot(2, 2, 2)
plt.title(f"Binary edges (95%ile)")
plt.imshow(bin_edges_display, cmap='gray')
plt.axis('off')

plt.subplot(2, 2, 3)
plt.title(f"Masked image")
plt.imshow(edges, cmap='gray')
plt.axis('off')

plt.subplot(2, 2, 4)
plt.title(f"Final output with threshold={threshold}")
plt.imshow(final, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()