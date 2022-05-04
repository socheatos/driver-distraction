import cv2

class Webcam:
    def __init__(self) -> None:
        self.cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)
        self.window_title = 'Test'

    def get_frame(self):
        self.ret, self.img = self.cap.read()
        self.gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        
    def show_frame(self):
        cv2.imshow(self.window_title,self.img)
    
    def show_point(self,point):
        cv2.circle(self.img,point,5, (0, 0, 255), -5)
        

# if __name__ == "__main__":
#     vid = Webcam()
#     while True: 
#         vid.get_frame()
#         vid.show_frame()
#         if cv2.waitKey(1) & 0xFF == 27:
#                 break       
#     vid.cap.release()
#     cv2.destroyAllWindows()