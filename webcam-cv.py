import cv2
import dlib
import utils
from threading import Thread
from queue import Queue

path = 'cascades\shape_predictor_68_face_landmarks.dat'

class Stream:
    def __init__(self) -> None:
        self.video = cv2.VideoCapture(0)    
    
    def stream(self):
        while True:
            self.image = self.video.read()[1]
            self.detect()

            cv2.imshow('Video',self.image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break


    def detect(self):
        face_det = dlib.get_frontal_face_detector()
        landmark = dlib.shape_predictor(path)

        # gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        gray =self.image
        faces = face_det(gray,0)
    
        for face in faces:
            shape = landmark(gray, face)
            shape = utils.shape_to_np(shape)

            self.face = cv2.rectangle(gray, (face.left(), face.top()),
                               (face.right(), face.bottom()), (0, 255, 0), 2)

            self.left_eye = (shape[36],shape[39])
            self.right_eye = (shape[42],shape[45])
            self.mouth = shape[62]

            self.image = cv2.circle(self.face,self.left_eye[0],2, (255, 0, 0), -1)
            self.image = cv2.circle(self.face,self.left_eye[1],2, (255, 0, 0), -1)
            self.image = cv2.circle(self.face,self.right_eye[0],2, (255, 0, 0), -1)
            self.image = cv2.circle(self.face,self.right_eye[1],2, (255, 0, 0), -1)
            self.image = cv2.circle(self.face,self.mouth,2, (255, 0, 0), -1)
        

if __name__ == "__main__":
    s = Stream()
    s.stream()