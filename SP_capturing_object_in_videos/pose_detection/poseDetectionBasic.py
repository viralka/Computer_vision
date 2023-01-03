from multiprocessing.spawn import import_main_path
import cv2 as cv
import mediapipe as mp 
import time

# cap = cv.VideoCapture(r'SP_capturing_object_in_videos\pose_detection\video\istockphoto-1174089637-640_adpp_is.mp4')
# cap = cv.VideoCapture(r'SP_capturing_object_in_videos\pose_detection\video\istockphoto-1166698457-640_adpp_is.mp4')

url = r"https://192.168.31.207:8080/video"
cap = cv.VideoCapture(url)

mpDwar = mp.solutions.drawing_utils

mpPose = mp.solutions.pose
pose = mpPose.Pose()
pt = 0
land = {}
while True:
    ret, frame = cap.read()
    
    # img1 = cv.resize(frame, (640, 480))
    img1 = frame
    img = cv.cvtColor(img1, cv.COLOR_BGR2RGB) 
    res = pose.process(img)
    # print(res.pose_landmarks)

    if res.pose_landmarks is not None:
        mpDwar.draw_landmarks(img1, res.pose_landmarks, mpPose.POSE_CONNECTIONS)

    # if res :
    #     # mpDwar.draw_landmarks(img, res.pose_Landmarks, mpPose.POSE_CONNECTIONS)
    #     # mpDwar.draw_skeleton(img, res.pose_Landmarks, mpPose.POSE_CONNECTIONS)
    #     # mpDwar.draw_skeleton(img, res.pose_Landmarks, mpPose.POSE_CONNECTIONS, thickness=2)
    #     # for id , lm in enumerate(res.pose_Landmarks):
    #     #     land[1]= lm
    #     pass
    
    ct = time.time()
    fps = int(1/(ct-pt))
    pt = ct

    cv.putText(img1, str(fps), (10,30), cv.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
    cv.imshow("video",img1)
    cv.waitKey(1)


