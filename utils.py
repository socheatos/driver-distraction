import cv2
import dlib
import numpy as np

path = 'cascades\shape_predictor_68_face_landmarks.dat'
face_det = dlib.get_frontal_face_detector()
landmark = dlib.shape_predictor(path)

def shape_to_np(shape, dtype="int"):
    # initialize
    coords = np.zeros((68, 2), dtype=dtype)
    # landmark to numpy
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords

def detect_landmark(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_det(gray)
    
    for face in faces:
        shape = landmark(gray, face)
        shape = shape_to_np(shape)

        left_eye = (shape[36],shape[39])
        right_eye = (shape[42],shape[45])
        mouth = shape[62]

        frame = cv2.circle(frame)

