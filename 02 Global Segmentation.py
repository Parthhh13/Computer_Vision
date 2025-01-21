import cv2
import numpy as np
import matplotlib.pyplot as plt

image=cv2.imread("lenna.jpg",cv2.IMREAD_GRAYSCALE)

def weighted_average(image):
    frequencies={}
    weighted_sum=0
    frequency_sum=0

    rows,cols=image.shape
    for row in range(rows):
        for col in range(cols):
            value=int(image[row][col])
            if value in frequencies:
                frequencies[value]+=1
            else:
                frequencies[value]=1
    for value in frequencies:
        weighted_sum+= value*frequencies[value]
        frequency_sum+=frequencies[value]

    result=weighted_sum/frequency_sum

    return result



def global_thresholding(image,initial, max_iter=100, tolerance=0.5):
    threshold = initial

    for iter in range(max_iter):
        iter+=1
        below=[]
        above=[]

        rows,cols=image.shape
        for row in range(rows):
            for col in range(cols):
                
                if image[row][col]<=threshold:
                    below.append(image[row][col])
                else:
                    above.append(image[row][col])

        if len(below)>0:
            meanb=np.mean(below)
        else:
            meanb=0
        
        if len(above)>0:
            meana=np.mean(above)
        else:
            meana=0
        
        new_threshold = (meanb + meana) // 2
        
        if abs(new_threshold - threshold) < tolerance:
            break
        
        threshold = new_threshold

    binary_image = np.where(image > threshold, 255, 0).astype(np.uint8)
    return iter,threshold, binary_image

# Display results

plt.figure(figsize=(5, 5))

# Original Image
plt.subplot(2, 2, 1)
plt.title("Original Image")
plt.imshow(image, cmap='gray')
plt.axis('off')

iter1,final_threshold1, binary_result1 = global_thresholding(image,weighted_average(image))
# Thresholded Binary Image
plt.subplot(2, 2, 2)
plt.title("Using weighted average")
plt.text(0.5, -0.1, f"Final threshold: {final_threshold1} in {iter1} iterations", 
         fontsize=10, ha='center', transform=plt.gca().transAxes)
plt.imshow(binary_result1, cmap='gray')
plt.axis('off')

iter2,final_threshold2, binary_result3 = global_thresholding(image,np.mean(image))
plt.subplot(2, 2, 3)
plt.title("Using mean")
plt.text(0.5, -0.1, f"final threshold: {final_threshold2} in {iter2} iterations", 
         fontsize=10, ha='center', transform=plt.gca().transAxes)
plt.imshow(binary_result3, cmap='gray')
plt.axis('off')

iter3,final_threshold3, binary_result4 = global_thresholding(image,np.var(image))
plt.subplot(2, 2, 4)
plt.title("Using variance")
plt.text(0.5, -0.1, f"final threshold: {final_threshold3} in {iter3} iterations", 
         fontsize=10, ha='center', transform=plt.gca().transAxes)
plt.imshow(binary_result4, cmap='gray')
plt.axis('off')


plt.show()