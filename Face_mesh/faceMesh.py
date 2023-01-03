import cv2 as cv
from cv2 import circle
import numpy as np
import time 
import os
import mediapipe as mp
from pytools import F, P 
import dev_Tools as dt

import cv2 as cv
import time 
import mediapipe as mp
import dev_Tools as dt

cap = cv.VideoCapture(r'D:\agnext\Agnext\OpenCv\Videos\face (1).mp4')
pTime= 0

mpFaceMesh = mp.solutions.face_mesh
mpDraw = mp.solutions.drawing_utils
faceDetection = mpFaceMesh.FaceMesh(max_num_faces=3)
drawSpecs = mpDraw.DrawingSpec(thickness = 1, circle_radius = 2, color = (0,255,0))

while True:
    ret, frame = cap.read()

    
    if ret:
        frame = dt.resizeimg(frame, 1000)
        imageRGB = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        result = faceDetection.process(frame)

        
        if result.multi_face_landmarks:
            for facelms in result.multi_face_landmarks:

                mpDraw.draw_landmarks(frame, facelms, mp.solutions.face_mesh.FACEMESH_CONTOURS, drawSpecs, drawSpecs)

                for id,lm in enumerate(facelms.landmark):
                    # print(lm)
                    ih , iw, ic = frame.shape
                    x,y = int(lm.x*iw), int(lm.y*ih)
                    print(id, x, y)

        
       
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



