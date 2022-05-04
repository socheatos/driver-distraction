from video import Webcam
from calibration import Video
from head import Head
import numpy as np
import cv2
import utils
import dlib
import math

#https://github.com/niconielsen32/ComputerVision/blob/master/headPoseEstimation.py

path = 'cascades\shape_predictor_68_face_landmarks.dat'
face_det = dlib.get_frontal_face_detector()
landmark = dlib.shape_predictor(path)

camera = Video()
camera.calibrate()

class LandMarks(Head):
    def __init__(self):
        super().__init__()

    def detect_features(self,face):
        shape = landmark(self.gray, face)
        self.shape = utils.shape_to_np(shape)
        self.face = ((face.left(), face.top()),(face.right(), face.bottom()))
        self.nose = self.shape[30]
        self.chin = self.shape[8]
        self.Leye_corner = self.shape[45]
        self.Reye_corner = self.shape[36]
        self.Lmouth_corner = self.shape[54]
        self.Rmouth_corner = self.shape[48]

    def detect_landmarks(self,verbose=False,all=False):
        faces = self.detect_face()
        for face in faces:
            self.detect_features(face)
            self.draw_landmarks(all)


    def draw_landmarks(self,all=False):
        pts = [self.nose,self.chin,self.Leye_corner, self.Reye_corner, self.Lmouth_corner,self.Rmouth_corner]

        for pt in pts:
            cv2.circle(self.img,pt,2, (255, 0, 0), -1)
        

        if all==True:
            for i in range(len(self.shape)):
                cv2.circle(self.img,self.shape[i],1, (255, 0, 0), -1)
        

def draw_annotation_box(img, r, t, i, d,color=(255, 255, 0), line_width=2):
    point_3d = []
    rear_size = 1 
    rear_depth = 0
    point_3d.append((-rear_size, -rear_size, rear_depth))
    point_3d.append((-rear_size, rear_size, rear_depth))
    point_3d.append((rear_size, rear_size, rear_depth))
    point_3d.append((rear_size, -rear_size, rear_depth))
    point_3d.append((-rear_size, -rear_size, rear_depth))

    front_size = img.shape[1]
    front_depth = front_size*2

    point_3d.append((-front_size, -front_size, front_depth))
    point_3d.append((-front_size, front_size, front_depth))
    point_3d.append((front_size, front_size, front_depth))
    point_3d.append((front_size, -front_size, front_depth))
    point_3d.append((-front_size, -front_size, front_depth))

    point_3d = np.array(point_3d, dtype=float).reshape(-1, 3)

    # Map to 2d img points
    (point_2d, _) = cv2.projectPoints(point_3d,
                                      r,
                                      t,
                                      i,
                                      d)
    point_2d = np.int32(point_2d.reshape(-1, 2))
    
    k = (point_2d[5] + point_2d[8])//2

    
    return(point_2d[2], k)

def trial(head,camera):
    font = cv2.FONT_HERSHEY_SIMPLEX 
    camera.distortion = np.zeros((4,1))

    model_points = np.array([   (0.0, 0.0, 0.0),             # Nose tip
                                (0.0, -330.0, -65.0),        # Chin
                                (-225.0, 170.0, -135.0),     # Left eye left corner
                                (225.0, 170.0, -135.0),      # Right eye right corne
                                (-150.0, -150.0, -125.0),    # Left Mouth corner
                                (150.0, -150.0, -125.0)      # Right mouth corner
                            ])

    image_points = np.array([head.nose,
                            head.chin,
                            head.Leye_corner,
                            head.Reye_corner,
                            head.Lmouth_corner,
                            head.Rmouth_corner], dtype='double')

    ret, rotation_vec, translation = cv2.solvePnP(model_points, image_points,camera.intrinsic, 
                                                camera.distortion, flags=cv2.SOLVEPNP_UPNP)
    rotation_vec = rotation_vec.reshape((3,))
    rotation,_ = cv2.Rodrigues(rotation_vec)
    angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rotation)
    # print(angles)
    x,y,z = angles[0], angles[1], angles[2]
    # nose_3d_projection, jacobian = cv2.projectPoints(nose_3d, rotation_vec, translation, camera.intrinsic, camera.distortion)

    cv2.putText(head.img, "x: " + str(np.round(x,2)), (500, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(head.img, "y: " + str(np.round(y,2)), (500, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(head.img, "z: " + str(np.round(z,2)), (500, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)



if __name__ == "__main__":
    vid = LandMarks()
    cam = Video()

    cam.calibrate()

    while True: 
        vid.get_frame()
        vid.detect_landmarks()

        trial(vid,cam)
        vid.show_frame()
        if cv2.waitKey(1) & 0xFF == 27:
            break
        
    vid.cap.release()
    cv2.destroyAllWindows()