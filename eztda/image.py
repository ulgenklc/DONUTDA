import numpy as np
import cv2

def enhance(img,alpha,beta):
    img = img.astype(np.float)
    img = img*alpha + beta
    img.clip(0,255,img)
    return img.astype(np.uint8)

def imread(fn):
    return cv2.imread(fn,0)
