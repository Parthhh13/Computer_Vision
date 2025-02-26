import numpy as np
import matplotlib.pyplot as plt
import cv2

# Load and denoise the image
<<<<<<< HEAD
image = cv2.imread(r"E:\Study Material\NIIT\6th Semester\Computer Vision\Lab\Crack.jpg", cv2.IMREAD_GRAYSCALE)
=======
image = cv2.imread(r"chair.jpg", cv2.IMREAD_GRAYSCALE)
>>>>>>> 976db7b74fb8d528c8cef10d7b0d669469b56a50
image=cv2.resize(image,(256,256))




#Find the binary edge matrix
def edge_thresholding(image):
    blur_image=cv2.GaussianBlur(image,(5,5),0)
    grad_X=cv2.Sobel(blur_image,cv2.CV_64F,1,0,ksize=3)
    grad_y=cv2.Sobel(blur_image,cv2.CV_64F,0,1,ksize=3)

    magnitude=(grad_X**2+grad_y**2)**0.5
    threshold_mag=0.5*np.max(magnitude)

    bin_edges=np.where(magnitude>threshold_mag,1,0)
    norm_magnitude = (magnitude / np.max(magnitude)) * 255
    bin_edges_display = np.where(norm_magnitude > (0.5 * 255), 255, 0)

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

        binary_image = np.where(image > 250, 255, 0)

        return optimal_threshold,binary_image

    threshold,_=otsu_thresholding(edges)

    final=np.where(image>threshold,255,0)
    return magnitude, bin_edges_display, edges, threshold, final

magnitude, bin_edges_display, edges, threshold, final = edge_thresholding(image)


plt.figure(figsize=(8, 6))

images=[image,magnitude,bin_edges_display,edges,final]
title=["Orignal Image","Sobel edges","Strong edges","Masked image","Final image"]

for i in range(len(images)):
    plt.subplot(2, 3, i+1)
    plt.title(title[i])
    plt.imshow(images[i], cmap='gray')
    plt.axis('off')

font = {'family': 'serif', 'color':  'darkred', 'weight': 'bold', 'size': 11}
plt.figtext(x=0.68, y=0.49,s= f"Optimal Threshold value = {threshold}",fontdict=font)
#plt.tight_layout()
plt.show()