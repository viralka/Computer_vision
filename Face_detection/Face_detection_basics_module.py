import cv2 as cv
import time 
import mediapipe as mp
import dev_Tools as dt



class FaceDetection():
    def __init__(self, minDetection = .5) :

        self.minDetection = minDetection
        self.mpFaceDetector = mp.solutions.face_detection
        self.mpDraw = mp.solutions.drawing_utils
        self.faceDetection = self.mpFaceDetector.FaceDetection(minDetection)
        # self.faceDetection.configure(min_confidence=self.minDetection)
        # self.faceDetection.start_async()

    def find_faces(self ,imageO, draw = True):
        image = cv.cvtColor(imageO, cv.COLOR_BGR2RGB)
        self.result = self.faceDetection.process(image)
        bboxs = []
        if self.result.detections:
            for id, detection in enumerate(self.result.detections):
                # print(id, detection)
                # mpDraw.draw_detection(frame, detection)
                bboxc  = detection.location_data.relative_bounding_box
                ih , iw , ic = image.shape
                bbox = int(bboxc.xmin*iw), int(bboxc.ymin*ih), \
                    int(bboxc.width*iw), int(bboxc.height*ih)
                
                bboxs.append([id,bbox, detection.score[0]])
                # print(bbox)
                if draw:
                    self.fancyDraw(imageO, bbox)
                    cv.putText(imageO, str(round(detection.score[0], 3)), (bbox[0], bbox[1]-20), cv.FONT_HERSHEY_COMPLEX,0.75,(255,255,255),1)

            return  [imageO,bbox]

 
    def fancyDraw(self,image, bbox, l = 30, t = 3):
        x,y,w,h = bbox
        # print(x,y,w,h)
        x1,y1 = x+w,y+h
        
        cv.rectangle(image, bbox ,(255,255,255), 1) 
        # Top left corner
        cv.line(image, (x,y), (x+l,y), (255,255,255), t)
        cv.line(image, (x,y), (x,y+l), (255,255,255), t)

        # bottom right corner
        cv.line(image, (x1,y1), (x1-l,y1), (255,255,255), t)
        cv.line(image, (x1,y1), (x1,y1-l), (255,255,255), t)


     
def main():
    url = r'https://192.168.27.219:8080/video'
    cap = cv.VideoCapture(0)
    pTime= 0
    detector = FaceDetection()
    while True:
        ret, frame = cap.read()

        if ret:
            frame = dt.resizeimg(frame, 1000)
            try:
                frame , bbox= detector.find_faces(frame)
            except:
                pass
            # print(bbox)

            cTime = time.time()
            fps  = 1/(cTime-pTime)
            pTime = cTime 

            cv.putText(frame, str(int(fps)), (10,70), cv.FONT_HERSHEY_COMPLEX,3,(255,0,255),3)
            # cv.rectangle(frame, bbox, (0,0,255), 2)



            cv.imshow('frame', frame)
            if cv.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
        pass


if __name__ == '__main__':
   
    main()
