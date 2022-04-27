import cv2
import dlib
import utils
from threading import Thread
from queue import Queue
#TODO threading to increase speed

path = 'cascades\shape_predictor_68_face_landmarks.dat'
face_det = dlib.get_frontal_face_detector()
landmark = dlib.shape_predictor(path)

class Video:
    def __init__(self) -> None:
        self.cap = cv2.VideoCapture(0)

    def detect(self):
        while True:
            _,self.img = self.cap.read()
            self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
            faces = face_det(self.img,0)
            
            for face in faces:
                shape = landmark(self.img, face)
                shape = utils.shape_to_np(shape)
                self.face = ((face.left(), face.top()),(face.right(), face.bottom()))
                self.left_eye = (shape[36],shape[39])
                self.right_eye = (shape[42],shape[45])
                self.mouth = shape[62]


                cv2.rectangle(self.img,self.face[0],
                               self.face[1], (0, 255, 0), 2)
                
                cv2.circle(self.img,self.left_eye[0],2, (255, 0, 0), -1)
                cv2.circle(self.img,self.left_eye[1],2, (255, 0, 0), -1)
                cv2.circle(self.img,self.right_eye[0],2, (255, 0, 0), -1)
                cv2.circle(self.img,self.right_eye[1],2, (255, 0, 0), -1)
                cv2.circle(self.img,self.mouth,2, (255, 0, 0), -1)

            cv2.imshow('test',self.img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    Video().detect()

        