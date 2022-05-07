import cv2
from cv2 import imwrite
import dlib
import utils
from video import Video
import numpy as np



path = 'cascades\shape_predictor_68_face_landmarks.dat'
face_det = dlib.get_frontal_face_detector()
landmark = dlib.shape_predictor(path)

class Detection():
    def __init__(self,video):
        super().__init__()
        self.video = video
        self.video.window_title = 'Landmark Detection'

    def detect_face(self):
        if self.video.img is not None:
            faces = face_det(self.video.img,0)
            return faces

    def detect_features(self,face):
        gray = cv2.cvtColor(self.video.img, cv2.COLOR_BGR2GRAY)
        shape = landmark(gray, face)
        shape = utils.shape_to_np(shape)
        self.face = ((face.left(), face.top()),(face.right(), face.bottom()))
        
        self.left_eye = shape[36:42]
        self.right_eye = shape[42:48] 
        self.mouth = [shape[48],shape[54],shape[51],shape[62],shape[66],shape[57]]
        self.nose = shape[30]
        self.chin = shape[8]

        self.Leye_corner = self.left_eye[0]
        self.Reye_corner = self.right_eye[4]
        self.Lmouth_corner = self.mouth[0]
        self.Rmouth_corner = self.mouth[1]
        return shape

    def detect_landmarks(self,show='HPE'):
        faces = self.detect_face()
        for face in faces:
            shape = self.detect_features(face)
            self.draw_landmarks(shape,show)
        
        
    def draw_landmarks(self,shape,show='HPE'):
        cv2.rectangle(self.video.img,self.face[0],
                               self.face[1], (0, 255, 0), 2)
        if show=='HPE':
            pts = [self.nose,self.chin,self.Leye_corner, self.Reye_corner, self.Lmouth_corner,self.Rmouth_corner]
            for pt in pts:
                cv2.circle(self.video.img,pt,2, (255, 0, 0), -1)

        elif show =='GAZE':
            feats = [self.left_eye, self.right_eye,self.mouth,[self.nose], [self.chin]]
            for pts in feats:
                for pt in pts:
                    cv2.circle(self.video.img,pt,2, (255, 0, 0), -1)

        else:
            for i in range(len(shape)):
                cv2.circle(self.video.img,shape[i],1, (255, 0, 0), -1)

                

    def get_head(self):
        while True: 
            self.get_frame()
            self.detect_landmarks()
            self.show_frame()
            if cv2.waitKey(1) & 0xFF == 27:
                # cv2.imwrite('data\calibration_data\head.jpg',self.img)
                break  
        self.cap.release()
        cv2.destroyAllWindows()
    

if __name__ == "__main__":
    video = Video('test.mp4')
    detector = Detection(video)
    while True: 
        video.get_frame()
        detector.detect_landmarks(show='GAZE')
        video.show_frame()
        if cv2.waitKey(1) & 0xFF == 27:
            break        
    video.vid.release()
    cv2.destroyAllWindows()

