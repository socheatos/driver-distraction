from video import Video
from detection import Detection
from calibration import Calibration
import cv2
import numpy as np
import utils
import dlib


class Pose():
    def __init__(self, detection, camera):
        self.detection = detection
        try:
            self.camera = camera.calibrate()
        except:
            self.camera = camera


    def estimate(self):
        font = cv2.FONT_HERSHEY_SIMPLEX 
        self.camera.distortion = np.zeros((4,1))

        model_points = np.array([   (0.0, 0.0, 0.0),             # Nose tip
                                    (0.0, -330.0, -65.0),        # Chin
                                    (-225.0, 170.0, -135.0),     # Left eye left corner
                                    (225.0, 170.0, -135.0),      # Right eye right corne
                                    (-150.0, -150.0, -125.0),    # Left Mouth corner
                                    (150.0, -150.0, -125.0)      # Right mouth corner
                                ])

        image_points = np.array([self.detection.nose,
                                self.detection.chin,
                                self.detection.Leye_corner,
                                self.detection.Reye_corner,
                                self.detection.Lmouth_corner,
                                self.detection.Rmouth_corner], dtype='double')

        ret, rotation_vec, translation = cv2.solvePnP(model_points, image_points,self.camera.intrinsic, 
                                                    self.camera.distortion, flags=cv2.SOLVEPNP_UPNP)
        rotation_vec = rotation_vec.reshape((3,))
        rotation,_ = cv2.Rodrigues(rotation_vec)
        angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rotation)
        self.x,self.y,self.z = angles[0], angles[1], angles[2]

        cv2.putText(self.camera.img, "x: " + str(np.round(self.x,2)), (500, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(self.camera.img, "y: " + str(np.round(self.y,2)), (500, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(self.camera.img, "z: " + str(np.round(self.z,2)), (500, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)


if __name__ == "__main__":
    vid = Video('test.mp4')
    detection = Detection(vid)
    pose = Pose(detection,vid)

    while True: 
        vid.get_frame()
        detection.detect_landmarks()
        pose.estimate()
        vid.show_frame()
        if cv2.waitKey(1) & 0xFF == 27:
            break
        
    vid.vid.release()
    cv2.destroyAllWindows()
