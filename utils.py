import cv2
import dlib
import numpy as np

def shape_to_np(shape, dtype="int"):
    # initialize
    coords = np.zeros((68, 2), dtype=dtype)
    # landmark to numpy
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords
