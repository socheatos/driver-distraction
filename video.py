import cv2

class Video:
    def __init__(self) -> None:
        self.cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)

    def get_frame(self):
        self.ret,self.img = self.cap.read()
        self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        
    def show_frame(self):
        cv2.imshow('Landmark Detection',self.img)

# if __name__ == "__main__":
#     vid = Video()
#     while True: 
#         vid.get_frame()
#         vid.show_frame()
#         if cv2.waitKey(1) & 0xFF == 27:
#                 break       
#     vid.cap.release()
#     cv2.destroyAllWindows()