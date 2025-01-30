import numpy as np
import matplotlib.pyplot as plt
import cv2


# Load and denoise the image
image = cv2.imread(r"E:\Study Material\NIIT\6th Semester\Computer Vision\Lab\uneven.jpg", cv2.IMREAD_GRAYSCALE)
#image=cv2.resize(image,(256,256))
image=cv2.GaussianBlur(image,(5,5),0)

def variable_thresholding(image,kernel_size,a,b):
    pad=kernel_size//2
    padded_image=np.pad(image,pad,mode='reflect')
    output_image=np.zeros_like(image)
    pad_rows,pad_cols=padded_image.shape

    for row in range(pad,pad_rows-pad):
        for col in range(pad,pad_cols-pad):
            kernel=padded_image[row-pad:row+pad+1,col-pad:col+pad+1]

            mean=np.mean(kernel)
            std=np.std(kernel)

            threshold=a*std+b*mean

            output_image[row - pad][ col - pad] = np.where(padded_image[row][col] > threshold, 255, 0)

    return output_image

otsu=np.where(image>151,255,0)
images=[image,otsu]
title=["Orignal Image","Otsus output"]
kernel_size,a,b=15,0.6,0.85
'''for i in range(10):
    a+=0.1'''
output=variable_thresholding(image,kernel_size,a,b)
images.append(output)
title.append(f'a,b={a,b}')

plt.figure(figsize=(8, 6))


for i in range(len(images)):
    plt.subplot(1, 3, i+1)
    plt.title(title[i])
    plt.imshow(images[i], cmap='gray')
    plt.axis('off')
plt.tight_layout()
plt.show()




    
