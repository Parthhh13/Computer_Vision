import numpy as np

# Load the image
'''image = cv2.imread("lenna.jpg", cv2.IMREAD_GRAYSCALE)
image=cv2.resize(image,(256,256))'''

def component_labeling(binary_image):
    rows, cols = binary_image.shape
    labels = np.zeros((rows, cols), dtype=int)
    label_counter = 1
    equivalence_dict = {}

    for row in range(rows):
        for col in range(cols):
            if binary_image[row][col]==1:

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
                    labels[row][col]=left_label 

                    if top_label!=left_label:
                        equivalence_dict[top_label]=left_label
   
    for row in range(rows):
        for col in range(cols):
            if labels[row][col] > 0:  # If pixel is labeled
                if labels[row][col] in equivalence_dict:
                     labels[row][col]=equivalence_dict[labels[row][col]]

    return labels


binary_image=np.array([
    [1,0,1,0,1],
    [1,0,1,0,1],
    [1,1,1,0,0],
    [0,0,0,0,1]
])
print(component_labeling(binary_image))



