import cv2 as cv
import time 
import mediapipe as mp
import dev_Tools as dt

url = r'https://192.168.27.219:8080/video'
cap = cv.VideoCapture(url)
# cap = cv.VideoCapture(0)
pTime= 0

mpFaceDetector = mp.solutions.face_detection
mpDraw = mp.solutions.drawing_utils
faceDetection = mpFaceDetector.FaceDetection()

while True:
    ret, frame = cap.read()

    
    if ret:
        # frame = dt.resizeimg(frame, 500)
        imageRGB = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        result = faceDetection.process(frame)
        

        if result.detections:
            for id, detection in enumerate(result.detections):
                # print(id, detection)
                # mpDraw.draw_detection(frame, detection)

                bboxc  = detection.location_data.relative_bounding_box
                ih , iw , ic = imageRGB.shape
                bbox = int(bboxc.xmin*iw), int(bboxc.ymin*ih), \
                    int(bboxc.width*iw), int(bboxc.height*ih)
                
                cv.rectangle(frame, bbox ,(255,255,255), 2) 
                cv.putText(frame, str(round(detection.score[0], 3)), (bbox[0], bbox[1]-20), cv.FONT_HERSHEY_COMPLEX,0.75,(255,255,255),1)
       
        
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
