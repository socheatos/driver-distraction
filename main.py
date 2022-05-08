from video import Video
from detection import Detection
from pose_estimation import Pose
from distraction_detection import DistractionScore
import cv2, time

def main():
    curr_time = 0 
    pst_time = 0 
    prev_time = 0 
    fps_lim = 11
    time_lim = 1/fps_lim

    # instantiate our objects
    vid = Video(0)
    detection = Detection(vid)
    pose = Pose()   
    scorer = DistractionScore(fps_lim,  PITCH_THRESH=20,YAW_THRESH=35, ROLL_THRESH=20, POSE_TIME_THRESH=3.0)

    while True: 
        delta = time.time()
        vid.get_frame()

        # if time passed is bigger or equal than frame time process the frame
        if delta >= time_lim:
            prev_time = time.time()
            # compute actual fps of webcam vid stearm
            curr_time = time.time()
            fps = 1.0/float(curr_time-pst_time)
            pst_time = curr_time
            cv2.putText(vid.img, "FPS: " + str(round(fps,0)), (50, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            detection.detect_landmarks(show='HPE')
            pitch, yaw, roll = pose.estimate(detection=detection, camera=vid)
            distracted = scorer.evaluate(pitch, yaw, roll)

            if distracted:
                cv2.putText(vid.img, "DISTRACTED", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2)
        
        vid.show_frame()
        if cv2.waitKey(1) & 0xFF == 27:
            break
          
    vid.vid.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
