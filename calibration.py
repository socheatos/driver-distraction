# https://www.ostirion.net/post/webcam-calibration-with-opencv-directly-from-the-stream
# checkerboard image https://static.wixstatic.com/media/dd11f9_a6f04e762a25471e878762136b992d24~mv2.png/v1/fill/w_1480,h_1485,al_c,q_90/dd11f9_a6f04e762a25471e878762136b992d24~mv2.webp

from video import Video
import cv2
import numpy as np
from numpy import savetxt
from numpy import genfromtxt
import os


CHECKERBOARD = (7,7)
MIN_POINTS = 50
STOP_CRITERIA = (cv2.TERM_CRITERIA_EPS +
            cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

objectp3d = np.zeros((1, CHECKERBOARD[0]
                      * CHECKERBOARD[1],
                      3), np.float32)

objectp3d[0, :, :2] = np.mgrid[0:CHECKERBOARD[0],
                               0:CHECKERBOARD[1]].T.reshape(-1, 2)

class Calibration(Video):
    def __init__(self) -> None:
        super().__init__()
        self.width = self.cap.get(3)
        self.height = self.cap.get(4)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.threeD_pts = []
        self.twoD_pts = []

    
    def get_frame(self):
        return super().get_frame()

    def find_corners(self):
        # Find the chess board corners
        # if desired number of corners are
        # found in the image then ret = true:

        self.ret, self.corners = cv2.findChessboardCorners(
                                    self.img, 
                                    CHECKERBOARD,
                                    cv2.CALIB_CB_ADAPTIVE_THRESH +
                                    cv2.CALIB_CB_FAST_CHECK +
                                    cv2.CALIB_CB_NORMALIZE_IMAGE
        )
        # If desired number of corners can be detected then,
        # refine the pixel coordinates and display
        # them on the images of checker board

        if self.ret == True:
            self.threeD_pts.append(objectp3d)
            self.corners2 = cv2.cornerSubPix(self.img,
                                self.corners,
                                CHECKERBOARD,
                                (-1,-1), STOP_CRITERIA)
            self.twoD_pts.append(self.corners2)

    def drawChessBoard(self):
        self.img =  cv2.drawChessboardCorners(self.img,
                                              CHECKERBOARD,
                                              self.corners, self.ret)

    def get_params(self):
        ret, matrix, distortion, r_vecs, t_vecs = cv2.calibrateCamera(
    self.threeD_pts, self.twoD_pts, self.img.shape[::-1], None, None) 

        mean_r = np.mean(np.asarray(r_vecs),axis=0)
        mean_t = np.mean(np.asarray(t_vecs),axis=0)

        if not os.path.exists('calibration_data'):
            os.makedirs('calibration_data')

        savetxt('calibration_data/rotation_vectors.csv', mean_r, delimiter=',')
        savetxt('calibration_data/translation_vectors.csv', mean_t, delimiter=',')
        savetxt('calibration_data/camera_matrix.csv', matrix, delimiter=',')
        savetxt('calibration_data/camera_distortion.csv', distortion, delimiter=',')
        
        print(" Camera matrix:")
        print(matrix)
        
        print("\n Distortion coefficient:")
        print(distortion)
        
        print("\n Rotation Vectors:")
        print(r_vecs)
        
        print("\n Translation Vectors:")
        print(t_vecs)

    def calibrate(self):
        while True: 
            self.get_frame()
            self.find_corners()
            self.drawChessBoard()
            self.show_frame()
            if (self.ret == True) and (len(self.twoD_pts) > MIN_POINTS):
                break
            if cv2.waitKey(1) & 0xFF == 27:
                    break       
        self.cap.release()
        cv2.destroyAllWindows()
        self.get_params()

if __name__ == "__main__":
    vid = Calibration()
    vid.calibrate()
    