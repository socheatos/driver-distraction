from video import Video
from detection import Detection
from pose_estimation import Pose
from distraction_detection import DistractionScore
import cv2, time

def main():
    # instantiate our objects
    vid = Video(0)
    detection = Detection(vid)
    pose = Pose()   
    scorer = DistractionScore(PITCH_THRESH=5,YAW_THRESH=5, ROLL_THRESH=20, POSE_FRAME_THRESH=50.0)

    distracted_count = 0
    count_frame = 0
    while True: 
        count_frame+=1
        fps = vid.vid.get(cv2.CAP_PROP_FPS)
        vid.get_frame()
    
        cv2.putText(vid.img, "Frame: " + str(count_frame), (50, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2
        detection.detect_landmarks(show='HPE')
        pitch, yaw, roll = pose.estimate(detection=detection,allpts=True)
        distracted = scorer.evaluate(pitch, yaw, roll)
        # print(distracted)
        if distracted:
            distracted_count+=1
        elif distracted_count > 0: 
            distracted_count-=fps
        if distracted_count>50:
            cv2.putText(vid.img, "DISTRACTED", (45, 55), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 5)
        
        vid.show_frame()
        if cv2.waitKey(1) & 0xFF == 27:
            break
          
    vid.vid.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
