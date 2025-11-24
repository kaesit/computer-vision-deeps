import torch.nn as nn
import numpy as np
import cv2


img = cv2.imread("image.jpg")
dim = (600, 600)
img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
cv2.namedWindow("Resized Image", cv2.WINDOW_NORMAL)
cv2.imshow("Resized Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

print(np.sqrt(16))

print("Computer Vision Deeps")