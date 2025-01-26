import numpy as np
import matplotlib.pyplot as plt
import cv2


# Load and denoise the image
image = cv2.imread("lenna.jpg", cv2.IMREAD_GRAYSCALE)
#image=cv2.resize(image,(256,256))


def edge_thresholding(image):
    blur_image=cv2.GaussianBlur(image,(5,5),0)
    grad_X=cv2.Sobel(blur_image,cv2.CV_64F,1,0,ksize=3)
    grad_y=cv2.Sobel(blur_image,cv2.CV_64F,0,1,ksize=3)

    magnitude=(grad_X**2+grad_y**2)**0.5
    threshold_mag=0.5*np.max(magnitude)

    bin_edges=np.where(magnitude>threshold_mag,1,0)

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
    return threshold,final


def partioning(image,w,h): 
    rows,cols=image.shape

    part_rows=rows//w
    part_cols=cols//h

    parts=[]
    for i in range(w):
        for j in range(h):
            part=image[i*part_rows:(i+1)*part_rows,j*part_cols:(j+1)*part_cols]
            parts.append(part)
    
    return parts,part_rows,part_cols

def reconstruction(parts, part_rows, part_cols, w, h):
    output = np.zeros((part_rows * w, part_cols * h), dtype=np.uint8)

    idx = 0
    for i in range(w):
        for j in range(h):
            output[i * part_rows:(i + 1) * part_rows, j * part_cols:(j + 1) * part_cols] = parts[idx]
            idx += 1

    return output


w, h = 3, 3  
parts, part_rows, part_cols = partioning(image, w, h)
images=[]
title=[]
for part in parts:
    thresh,bin=edge_thresholding(part)
    images.append(bin)
    title.append(thresh)

output=reconstruction(images,part_rows,part_cols,w,h)

for i in range(len(images)):
    plt.subplot(w, h, i+1)
    plt.title(f'Threshold ={title[i]}')
    plt.imshow(images[i], cmap='gray')
    #plt.axis('off')
plt.tight_layout()
plt.show()

display=[image,output]
titles=["Input Image","Output Image"]
for i in range(len(display)):
    plt.subplot(1, 2, i+1)
    plt.title(titles[i])
    plt.imshow(display[i], cmap='gray')
    plt.axis('off')
plt.tight_layout()
plt.show()