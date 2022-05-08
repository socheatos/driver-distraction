from statistics import mode

from cv2 import CAP_PROP_FPS
from video import Video
from detection import Detection
from calibration import Calibration
import cv2
import numpy as np
import utils
import dlib
import math


class Pose():
    def __init__(self, camera):
        self.axis = np.float32([[200, 0, 0], [0, 200, 0], [0, 0, 200], [0, 0, 0]]).reshape(-1, 3)
        try:
            self.camera = camera.calibrate()
        except:
            self.camera = camera
    def isRotationMatrix(self,R) :
        Rt = np.transpose(R)
        shouldBeIdentity = np.dot(Rt, R)
        I = np.identity(3, dtype = R.dtype)
        n = np.linalg.norm(I - shouldBeIdentity)
        return n < 1e-6
 
    def rotationMatrixToEulerAngles(self,R) :
        assert(self.isRotationMatrix(R))
        sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
        singular = sy < 1e-6
        if  not singular :
            x = math.atan2(R[2,1] , R[2,2])
            y = math.atan2(-R[2,0], sy)
            z = math.atan2(R[1,0], R[0,0])
        else :
            x = math.atan2(-R[1,2], R[1,1])
            y = math.atan2(-R[2,0], sy)
            z = 0
        return np.array([x, y, z])


    def draw_axes(self, imgpt, projpt):
        self.camera.img = cv2.line(self.camera.img, imgpt, tuple(
            projpt[0].ravel().astype(int)), (255, 0, 0), 3)
        self.camera.img = cv2.line(self.camera.img, imgpt, tuple(
            projpt[1].ravel().astype(int)), (0, 255, 0), 3)
        self.camera.img = cv2.line(self.camera.img, imgpt, tuple(
            projpt[2].ravel().astype(int)), (0, 0, 255), 3)

    def estimate(self,detection,allpts=False):
        font = cv2.FONT_HERSHEY_SIMPLEX 
        self.camera.distortion = np.zeros((4,1))

        model_points = np.array([   (0.0, 0.0, 0.0),             # Nose tip
                                    (0.0, -330.0, -65.0),        # Chin
                                    (-225.0, 170.0, -135.0),     # Left eye left corner
                                    (225.0, 170.0, -135.0),      # Right eye right corne
                                    (-150.0, -150.0, -125.0),    # Left Mouth corner
                                    (150.0, -150.0, -125.0)      # Right mouth corner
                                ])

        image_points = np.array([detection.nose,
                                detection.chin,
                                detection.Leye_corner,
                                detection.Reye_corner,
                                detection.Lmouth_corner,
                                detection.Rmouth_corner], dtype='double')

        if allpts:
            model_points, image_points = utils.get_all_68_pts(detection=detection)
        
        _, rotation, translation = cv2.solvePnP(model_points, image_points,self.camera.intrinsic, 
                                                    self.camera.distortion, flags=cv2.SOLVEPNP_UPNP)
       
        # refine both rotation and translation 
        rotation, translation =  cv2.solvePnPRefineLM(model_points, image_points, self.camera.intrinsic,
                                                        self.camera.distortion, rotation, translation)
        
        # make nose the axis
        nose = int(detection.nose[0]), int(detection.nose[1])
        nose_end_2D,_ = cv2.projectPoints(self.axis,rotation,translation, self.camera.intrinsic, self.camera.distortion)
        self.draw_axes(nose, nose_end_2D)

        rotation = rotation.reshape((3,))
        rotation,_ = cv2.Rodrigues(rotation) 

        # angles = self.rotationMatrixToEulerAngles(rotation)       

        # print(rotation)
        angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rotation)
        self.pitch,self.yaw,self.roll = angles[0], angles[1], angles[2]

        cv2.putText(self.camera.img, "pitch: " + str(np.round(self.pitch,2)), (50, 375), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.putText(self.camera.img, "yaw: " + str(np.round(self.yaw,2)), (50, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.putText(self.camera.img, "roll: " + str(np.round(self.roll,2)), (50, 425), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        return self.pitch, self.yaw,self.roll        

if __name__ == "__main__":
    vid = Video(0)
    detection = Detection(vid)
    pose = Pose(vid)   

    while True: 
        vid.get_frame()
        detection.detect_landmarks(show='HPE')
        pitch,yaw,roll = pose.estimate(detection=detection,allpts=False)
        
        vid.show_frame()
        if cv2.waitKey(1) & 0xFF == 27:
            break
        
    vid.vid.release()
    cv2.destroyAllWindows()
