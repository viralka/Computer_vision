import cv2 as cv
import mediapipe as mp 
import time 
import image_opener as io

# load the video from the phone 
# url = r"https://192.168.31.207:8080/video" # change everythime we have to jump ip
# cap = cv.VideoCapture(url)

cap = cv.VideoCapture(0)

# using inbuild methord
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

# for frame per second
pTime = 0
cTime = 0

while(True):

    # reading the frame from the camera
    camera, frame = cap.read()
    frame = io.resizeimg(frame, 1000)

    if frame is not None:
        imgRgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        result = hands.process(imgRgb)
        # print(result.multi_hand_landmarks)
        if result.multi_hand_landmarks:
            for handlms in result.multi_hand_landmarks:
                for id, lm in enumerate(handlms.landmark):
                    
                    # print(id,lm)
                    h, w,c = frame.shape
                    cx , cy = int(lm.x*w), int(lm.y*h)
                    print(id,cx,cy)

                    if id == 4: # each id is equal some specific point in the hand
                        cv.circle(frame, (cx,cy),25,(255,0,255), cv.FILLED)

                mpDraw.draw_landmarks(frame,handlms, mpHands.HAND_CONNECTIONS)

        cTime = time.time()
        fps  = 1/(cTime-pTime)
        pTime = cTime 

        cv.putText(frame, str(int(fps)), (10,70), cv.FONT_HERSHEY_COMPLEX,3,(255,0,255),3)
        cv.imshow("Frame", frame)
    q = cv.waitKey(1) 
    if q==ord("q"):
        break



cv.destroyAllWindows()
