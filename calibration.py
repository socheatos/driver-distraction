# https://www.ostirion.net/post/webcam-calibration-with-opencv-directly-from-the-stream
# checkerboard image https://static.wixstatic.com/media/dd11f9_a6f04e762a25471e878762136b992d24~mv2.png/v1/fill/w_1480,h_1485,al_c,q_90/dd11f9_a6f04e762a25471e878762136b992d24~mv2.webp
# https://docs.opencv.org/3.4/d4/d94/tutorial_camera_calibration.html

from video import Video
import cv2
import numpy as np
from numpy import savetxt
from numpy import genfromtxt
import os
import PIL.Image
import PIL.ExifTags

test_path = r'data\test.jpg'
param_path = r'data\calibration_data'

CHECKERBOARD = (7,7)
MIN_POINTS = 50
STOP_CRITERIA = (cv2.TERM_CRITERIA_EPS +
            cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

objectp3d = np.zeros((1, CHECKERBOARD[0]
                      * CHECKERBOARD[1],
                      3), np.float32)

objectp3d[0, :, :2] = np.mgrid[0:CHECKERBOARD[0],
                               0:CHECKERBOARD[1]].T.reshape(-1, 2)

class Parameters(Video):
    def __init__(self):
        super().__init__()
        self.width = self.cap.get(3)
        self.height = self.cap.get(4)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.threeD_pts = []
        self.twoD_pts = []
        self.window_title = 'Calibrating...'
        try:
            self.get_testIMG_params(test_path)
        except:  
            raise Exception("Please take image using your webcam name as 'test.jpg' and place it in dataset folder!")
        # try:
        #     self.get_camParameters(param_path)
        #     print('Found all the parameters!')
        # except:
        #     self.calibrate()
        self.calibrate()


    
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
            self.drawChessBoard()

    def drawChessBoard(self):
        self.img =  cv2.drawChessboardCorners(self.img,
                                              CHECKERBOARD,
                                              self.corners, self.ret)

    def find_params(self):
        print('Calculating parameters...')
        ret, self.intrinsic, self.distortion, self.r_vecs, self.t_vecs = cv2.calibrateCamera(
            self.threeD_pts, self.twoD_pts, self.img.shape[::-1], None, None) 

        mean_r = np.mean(np.asarray(self.r_vecs),axis=0)
        self.translation = np.mean(np.asarray(self.t_vecs),axis=0)
        self.rotation = cv2.Rodrigues(mean_r)[0]

        if not os.path.exists('data/calibration_data'):
            os.makedirs('data/calibration_data')

        savetxt('data/calibration_data/translation.csv', self.translation, delimiter=',')
        # savetxt('data/calibration_data/rotation_vectors.csv', self.r_vecs, delimiter=',')
        # savetxt('data/calibration_data/translation_vectors.csv', self.t_vecs, delimiter=',')
        savetxt('data/calibration_data/camera_matrix.csv', self.intrinsic, delimiter=',')
        savetxt('data/calibration_data/camera_distortion.csv', self.distortion, delimiter=',')
        savetxt('data/calibration_data/rotation_matrix.csv', self.rotation, delimiter=',')
        
        self.param_explode()
        print('Found all the parameters!')
    
    def get_testIMG_params(self, path):
        test_image = PIL.Image.open(path)
        exif = { PIL.ExifTags.TAGS[k]: v
                for k, v in test_image._getexif().items()
                if k in PIL.ExifTags.TAGS}
        self.focalLength = exif['FocalLength']
        self.native_X = exif['ImageWidth']
        self.native_Y = exif['ImageLength']


    def get_camParameters(self,path):
        self.intrinsic = genfromtxt(os.path.join(path,'camera_matrix.csv'),delimiter=',')
        self.rotation = genfromtxt(os.path.join(path,'rotation_matrix.csv'),delimiter=',')
        self.translation = genfromtxt(os.path.join(path,'translation.csv'),delimiter=',')
        # self.r_vecs =  genfromtxt(os.path.join(path,'rotation_vectors.csv'),delimiter=',')
        # self.t_vecs = genfromtxt(os.path.join(path,'translation_vectors.csv'),delimiter=',')
        self.param_explode()

    def param_explode(self):
        self.ox, self.oy = self.intrinsic[:2,-1]
        self.fx, self.fy = self.intrinsic[0][0], self.intrinsic[1][1] # focal length * scaling factor
        self.mx, self.my = self.fx/self.focalLength, self.fy/self.focalLength

    def calibrate(self):
        while True: 
            self.get_frame()
            self.find_corners()
            self.show_frame()
            if (self.ret == True) and (len(self.twoD_pts) > MIN_POINTS):
                cv2.imwrite('calibration_data\calib.jpg',self.img)
                break
            if cv2.waitKey(1) & 0xFF == 27:
                break       
        self.cap.release()
        cv2.destroyAllWindows()
        self.find_params()
