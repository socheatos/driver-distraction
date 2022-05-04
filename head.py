import cv2
from cv2 import imwrite
import dlib
import utils
from calibration import Video
from video import Webcam
import numpy as np
# from mtcnn.mtcnn import MTCNN

#TODO: center of projection (0,0,0) at image center

path = 'cascades\shape_predictor_68_face_landmarks.dat'
face_det = dlib.get_frontal_face_detector()
landmark = dlib.shape_predictor(path)

class Head(Webcam):
    def __init__(self):
        super().__init__()
        self.window_title = 'Landmark Detection'


    def detect_face(self):
        faces = face_det(self.img,0)
        return faces

    def detect_features(self,face):
        shape = landmark(self.gray, face)
        self.shape = utils.shape_to_np(shape)
        self.face = ((face.left(), face.top()),(face.right(), face.bottom()))
        self.left_eye = (int((self.shape[36][0]+self.shape[39][0])/2),int((self.shape[36][1]+self.shape[39][1])/2))
        self.right_eye = (int((self.shape[42][0]+self.shape[45][0])/2),int((self.shape[42][1]+self.shape[45][1])/2))
        self.mouth = self.shape[62]

    def detect_landmarks(self,verbose=False,all=False):
        faces = self.detect_face()
        for face in faces:
        
            self.detect_features(face)
            self.cx = (self.left_eye[0]+self.right_eye[0]+self.mouth[0])//3
            self.cy = (self.left_eye[1]+self.right_eye[1]+self.mouth[1])//3
            # self.generate_ellipse()
            self.draw_landmarks(all)


            if verbose == True:
                print(self.face, self.left_eye,self.right_eye,self.mouth)
        
    def draw_landmarks(self,all=False):
        cv2.rectangle(self.img,self.face[0],
                               self.face[1], (0, 255, 0), 2)

        cv2.circle(self.img,self.left_eye,2, (255, 0, 0), -1)
        cv2.circle(self.img,self.right_eye,2, (255, 0, 0), -1)
        cv2.circle(self.img,self.mouth,2, (255, 0, 0), -1)
        cv2.circle(self.img,(self.cx,self.cy),2, (255, 0, 0), -1)

        if all==True:
            for i in range(len(self.shape)):
                cv2.circle(self.img,self.shape[i],1, (255, 0, 0), -1)
        #principal point - where COP (0,0,0)
        # cv2.circle(self.img,(int(self.params.ox),int(self.params.oy)),2, (255, 0, 0), -1)
        

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
    
    
    # def generate_ellipse(self):
    #     P = np.array([self.left_eye,self.right_eye, self.mouth]) - np.array([self.cx,self.cy]) 
    #     sigma = np.dot(P.T,P)/3
        
    #     # radius of ellipse
    #     r = sigma[0][0] + sigma[1][1]+ np.sqrt((sigma[0][0] - sigma[1][1])**2+4*sigma[0][1]**2)
    #     self.pz = self.params.focalLength*(40/r)
    #     self.px = (self.cx-self.params.ox)*(self.pz/self.params.focalLength)
    #     self.py = (self.cy-self.params.oy)*(self.pz/self.params.focalLength)
        
    #     self.p = np.array([self.px, self.py, self.pz])
        
    #     self.getRoll()
        
    # def getRoll(self):
    #     roll = np.arctan((self.right_eye[1]-self.left_eye[1])/(self.right_eye[0]-self.left_eye[0]))
    #     # print(self.p,roll)

# if __name__ == "__main__":
#     vid = Head()
#     while True: 
#         vid.get_frame()
#         vid.detect_landmarks()
#         vid.show_frame()
#         if cv2.waitKey(1) & 0xFF == 27:
#             # imwrite('data\calibration_data\head.jpg',vid.img)
#             break
        
#     vid.cap.release()
#     cv2.destroyAllWindows()

