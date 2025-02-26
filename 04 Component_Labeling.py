import numpy as np
import cv2
import matplotlib.pyplot as plt

# Load the image
image = cv2.imread("chair.jpg", cv2.IMREAD_GRAYSCALE)

#image=cv2.resize(image,(100,100))

def component_labeling(binary_image):
    rows, cols = binary_image.shape
    labels = np.zeros((rows, cols), dtype=int)
    label_counter = 1
    equivalence_dict = {}

    for row in range(rows):
        for col in range(cols):
            if binary_image[row][col]==255:

                top_label=labels[row-1][col] if row>0 else 0
                left_label=labels[row][col-1] if col>0 else 0

                if top_label==0 and left_label==0:
                    labels[row][col]=label_counter
                    label_counter+=1

                elif top_label!=0 and left_label==0:
                    labels[row][col]=top_label

                elif top_label==0 and left_label!=0:
                    labels[row][col]=left_label

                elif top_label!=0 and left_label!=0:
                    min_label=min(top_label,left_label)
                    max_label=max(top_label,left_label)

                    labels[row][col]=min_label 

                    if min_label!=max_label:
                        equivalence_dict[max_label]=min_label
    def find_root(label):
        while label in equivalence_dict:
            label=equivalence_dict[label]
        return label
     
    for label in equivalence_dict.keys():
        equivalence_dict[label]=find_root(label)

    for row in range(rows):
        for col in range(cols):
            if labels[row][col] > 0:  # If pixel is labeled
                labels[row][col]=find_root(labels[row][col])
    
    distinct=np.unique(labels[labels>0])

    return labels,len(distinct)



plt.figure(figsize=(8, 6))

# Original Image
plt.subplot(1, 3, 1)
plt.title("Original Image")
plt.imshow(image, cmap='gray')
plt.axis('off')


thresh=90
bin = np.where(image > thresh, 255, 0).astype(np.uint8)
bin2=cv2.GaussianBlur(bin,(25,25),0)
plt.subplot(1, 3, 2)
plt.title(f"Segmented Image T={thresh}")
plt.imshow(bin, cmap='gray')
plt.axis('off')

final,labels=component_labeling(bin2)

# Original Image
plt.subplot(1, 3, 3)
plt.title(f"Labeled Image , No. of objects ={labels}")
plt.imshow(final, cmap='nipy_spectral')
plt.axis('off')

plt.tight_layout()
plt.show()



