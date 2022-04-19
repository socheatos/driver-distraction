import cv2
from sklearn.preprocessing import scale
# https://www.mygreatlearning.com/blog/viola-jones-algorithm/
# https://medium.com/0xcode/the-viola-jones-face-detection-algorithm-3eb09055cfc2


path = ['haarcascade_frontalface_default.xml','haarcascade_eye_tree_eyeglasses.xml']

class ViolaJones():
    def __init__(self,path,scaleFactor = 1.3, minNeighbors =3, minSize=(60,60)):
        self.face_cascade = cv2.CascadeClassifier(path[0])
        self.eye_cascade = cv2.CascadeClassifier(path[1])
        self.scaleFactor = scaleFactor
        self.minNeighbros = minNeighbors
        self.minSize = minSize


    def detect_faces(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor = self.scaleFactor,
                minNeighbors = self.minNeighbros,
                minSize = self.minSize
                )
        

    
    def draw_boxes(self,frame):
        for (x, y, w, h) in self.faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            faceROI = frame[y:y+h,x:x+w]
            eyes = self.eye_cascade.detectMultiScale(faceROI)
            i =0
            for (x2, y2, w2, h2) in eyes:
                #TODO get the coordinates of the eyes 
                               
                #left corner of the eyes
                frame = cv2.circle(frame, (x + x2 , (y + y2+h2//2)),2 , (255, 0, 0), -1)

                #right of the eyes
                frame = cv2.circle(frame, (x + x2+w2 , (y + y2+h2//2)),2 , (255, 0, 0), -1)
