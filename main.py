import torch.nn as nn
import numpy as np
import cv2
from utils.image_viewer import*
from utils.open_webcam import *

open_image("img/image.jpg", "My Image", 600, 400)
open_webcam(0, "Webcam", 640, 480)

print(np.sqrt(16))

print("Computer Vision Deeps")