import cv2 as cv
import time 
import numpy as np
import handTrakingModule as htm
import math

#for volume
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume


# for volume control using external libraryies developend by fine gentlemen AndreMiras
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
# volume.GetMute()
# volume.GetMasterVolumeLevel()

# volume.SetMasterVolumeLevel(0, None)


vol_min=volume.GetVolumeRange()[1]
vol_max= volume.GetVolumeRange()[0]



url = r"https://192.168.27.219:8080/video"
cap = cv.VideoCapture(0)
ptime= 0


detector = htm.handDetector(detectionCon=0.7)
vol =0
volBar = 0
while(True):
    camera, frame = cap.read()
    frame = cv.resize(frame,(1280,720),fx=0,fy=0, interpolation = cv.INTER_CUBIC)

    # for frame rate
    ctime = time.time()
    fps = 1/(ctime-ptime)
    ptime = ctime


    if frame is not None:

        frame = detector.findHands(frame)
        lmList = detector.findPosition(frame, draw=False)

        if len(lmList) != 0:
            x1, y1 = lmList[4][1], lmList[4][2]
            x2, y2 = lmList[8][1], lmList[8][2]

            cx, cy = int((x1+x2)/2), int((y1+y2)/2)

            cv.circle(frame, (x1, y1),15, (255, 0, 255), cv.FILLED)
            cv.circle(frame, (x2, y2),15, (255, 0, 255), cv.FILLED)
            cv.circle(frame, (cx,cy),15, (255, 0, 255), cv.FILLED)
            cv.line(frame, (x1, y1),(x2,y2),(225,0,255), thickness=3)
            length = math.hypot(x2-x1, y2-y1)
            # print(length)
            vol = np.interp(length, [50, 350], [vol_max, vol_min])
            # vol = vol_max - vol
            # print(vol)
            volBar = np.interp(length, [50, 350], [400, 150])   

            try:
                volume.SetMasterVolumeLevel(vol, None)  
            except:
                print("error")   

            if length < 50:
                cv.circle(frame, (cx,cy),15, (0, 255, 0), cv.FILLED)
        
            cv.rectangle(frame, (50,150), (85, 400), (32,32,38),3)
            cv.rectangle(frame, (50,int(volBar)), (85, 400), (32,32,38),cv.FILLED)

        # writing frame on the screen 
        frame = cv.resize(frame,(1280,720),fx=0,fy=0, interpolation = cv.INTER_CUBIC)
        cv.putText(frame, "FPS: " + str(int(fps)), (10,50), cv.FONT_HERSHEY_COMPLEX,1,(255,0,255),3)
        cv.imshow("Frame", frame)


    q = cv.waitKey(1)
    if q==ord("q"):
        break
cv.destroyAllWindows()