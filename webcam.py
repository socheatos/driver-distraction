#source: https://scribles.net/showing-video-image-on-tkinter-window-with-opencv/

import tkinter as tk
from PIL import Image, ImageTk
import cv2
import dlib
import utils

path = ['cascades\haarcascade_frontalface_default.xml','cascades\haarcascade_eye_tree_eyeglasses.xml','cascades\shape_predictor_68_face_landmarks.dat']
class MainWindow():
    
    def __init__(self, window, cap):
        self.window = window
        self.cap = cap
        self.width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self.interval = 50 # Interval in ms to get the latest frame

        # Create canvas for image
        self.canvas = tk.Canvas(self.window, width=self.width, height=self.height)
        self.canvas.grid(row=0, column=0)
        # Update image on canvas
        self.update_image()

    def update_image(self):
        # call violajones
        # VJ = ViolaJones(path)
        # Get the latest frame and convert image format
        self.image = cv2.cvtColor(self.cap.read()[1], cv2.COLOR_BGR2RGB) # to RGB
        self.detect_landmark()
        self.image = Image.fromarray(self.image) # to PIL format
        self.image = ImageTk.PhotoImage(self.image) # to ImageTk format

        # Update image
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.image)
        # Repeat every 'interval' ms
        self.window.after(self.interval, self.update_image)

    def detect_landmark(self):
        face_det = dlib.get_frontal_face_detector()
        landmark = dlib.shape_predictor(path[2])

        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        faces = face_det(gray)
    
        for face in faces:
            shape = landmark(gray, face)
            shape = utils.shape_to_np(shape)

            left_eye = (shape[36],shape[39])
            right_eye = (shape[42],shape[45])
            mouth = shape[62]

            self.image = cv2.circle(self.image,left_eye[0],2, (255, 0, 0), -1)
            self.image = cv2.circle(self.image,left_eye[1],2, (255, 0, 0), -1)
            self.image = cv2.circle(self.image,right_eye[0],2, (255, 0, 0), -1)
            self.image = cv2.circle(self.image,right_eye[1],2, (255, 0, 0), -1)
            self.image = cv2.circle(self.image,mouth,2, (255, 0, 0), -1)




if __name__ == "__main__":
    root = tk.Tk()
    MainWindow(root, cv2.VideoCapture(0))
    root.mainloop()