import cv2
import numpy as np

img1 = cv2.imread('5img1.jpeg', 0)
img2 = cv2.imread('5img2.jpeg', 0)

#--- take the absolute difference of the images ---
res = cv2.absdiff(img1, img2)

#--- convert the result to integer type ---
res = res.astype(np.uint8)

percentage = abs((np.count_nonzero(res) * 100)/ res.size - 100)

print(percentage)