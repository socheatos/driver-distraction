import cv2
import dlib
import utils
from video import Video

#TODO: calibration - estimate focal length and face size

path = 'cascades\shape_predictor_68_face_landmarks.dat'
face_det = dlib.get_frontal_face_detector()
landmark = dlib.shape_predictor(path)

class Head(Video):
    def __init__(self) -> None:
        super().__init__()

    def detect_face(self):
        faces = face_det(self.img,0)
        return faces

    def detect_features(self,face):
        shape = landmark(self.img, face)
        shape = utils.shape_to_np(shape)
        self.face = ((face.left(), face.top()),(face.right(), face.bottom()))
        self.left_eye = (shape[36],shape[39])
        self.right_eye = (shape[42],shape[45])
        self.mouth = shape[62]

    def detect_landmarks(self,verbose=False):
        faces = self.detect_face()
        for face in faces:
            self.detect_features(face)
            self.draw_landmarks()

            if verbose == True:
                print(self.face, self.left_eye,self.right_eye,self.mouth)
        
    def draw_landmarks(self):
        cv2.rectangle(self.img,self.face[0],
                               self.face[1], (0, 255, 0), 2)
                
        cv2.circle(self.img,self.left_eye[0],2, (255, 0, 0), -1)
        cv2.circle(self.img,self.left_eye[1],2, (255, 0, 0), -1)
        cv2.circle(self.img,self.right_eye[0],2, (255, 0, 0), -1)
        cv2.circle(self.img,self.right_eye[1],2, (255, 0, 0), -1)
        cv2.circle(self.img,self.mouth,2, (255, 0, 0), -1)

if __name__ == "__main__":
    vid = Head()
    while True: 
        vid.get_frame()
        vid.detect_landmarks()
        vid.show_frame()
        if cv2.waitKey(1) & 0xFF == 27:
                break
        
    vid.cap.release()
    cv2.destroyAllWindows()