from video import Video
from pose_estimation import Pose
from detection import Detection
import cv2
# https://github.com/e-candeloro/Driver-State-Detection/blob/master/driver_state_detection/Driver_State_Detection.py

class DistractionScore():
    def __init__(self,fps_lim, ROLL_THRESH, PITCH_THRESH , YAW_THRESH, POSE_TIME_THRESH = 4.0):
        
        self.fps = fps_lim          # upper frame rate of video stream considered
        self.delta_time_frame = 1/fps_lim
        self.prev_time = 0

        self.ROLL_THRESH = ROLL_THRESH
        self.PITCH_THRESH = PITCH_THRESH
        self.YAW_THRESH = YAW_THRESH

        self.pose_time_thresh = POSE_TIME_THRESH    # max time allowable for consecutive distracted headpose 
        self.pose_act_tresh = self.pose_time_thresh / self.delta_time_frame
        self.pose_counter = 0

    def evaluate(self, pitch, roll, yaw):
        distracted = False 

        if self.pose_counter >= self.pose_act_tresh:
            distracted=True
        
        if ((abs(roll) < self.ROLL_THRESH) or (abs(pitch) > self.PITCH_THRESH) or (abs(yaw) > self.YAW_THRESH)):
            if not distracted:
                self.pose_counter += 1
                
        elif self.pose_counter > 0:
            self.pose_counter -= 1

        return distracted


        