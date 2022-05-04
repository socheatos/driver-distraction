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

def distance(p1, p2):
    d = np.sqrt((p1[0]-p2[0])**2+(p1[1]-p2[1])**2)
    return d

def getBoxCoord(box):
    # self.face = ((face.left(), face.top()),(face.right(), face.bottom()))
    left_top = box[0]
    right_bottom = box[1]

    right_top = (right_bottom[0],left_top[1])
    left_bottom = (left_top[0],right_bottom[1])

    return left_top,right_top, left_bottom, right_bottom

