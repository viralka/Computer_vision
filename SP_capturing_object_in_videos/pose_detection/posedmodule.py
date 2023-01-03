from multiprocessing.spawn import import_main_path
import cv2 as cv
from cv2 import LMEDS
import mediapipe as mp 
import time

    
class poser():

        def __init__(self,
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            enable_segmentation=False,
            smooth_segmentation=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5):

            self.mode = static_image_mode
            self.complexity = model_complexity
            self.landmarks = smooth_landmarks
            self.segmentation = enable_segmentation
            self.smo_segmentation = smooth_segmentation
            self.detection = min_detection_confidence
            self.tracking = min_tracking_confidence


            # self.pose_graph = mp.PoseGraph()
            self.mpDwar = mp.solutions.drawing_utils
            self.mpPose = mp.solutions.pose
            self.pose = self.mpPose.Pose(self.mode, self.complexity, self.landmarks, self.segmentation, self.smo_segmentation, self.detection, self.tracking)

        def findPose(self, img, draw = True):

            img1 = cv.cvtColor(img, cv.COLOR_BGR2RGB) 
            self.res = self.pose.process(img1)
            if draw:
                if self.res.pose_landmarks is not None:
                    self.mpDwar.draw_landmarks(img, self.res.pose_landmarks, self.mpPose.POSE_CONNECTIONS)

            return img


        def findPositions(self, img, draw = True):
            lmList = []
            if self.res.pose_landmarks is not None:
                for id , lm in enumerate(self.res.pose_landmarks.landmark):
                    h, w, c = img.shape

                    cx, cy = int(lm.x * w), int(lm.y * h)
                    lmList.append((id, cx, cy))
                    # print((id, cx, cy))

                    if draw is True:
                        cv.circle(img, (cx, cy), 3, (225,0,0), cv.FILLED)

            return lmList

            

def video(url = None):
    if url is not None:
        return cv.VideoCapture(url)

    else:
        return cv.VideoCapture(r'SP_capturing_object_in_videos\pose_detection\video\istockphoto-1174089637-640_adpp_is.mp4')



def main():
    cap = video()
    pt = 0
    detector = poser()
    # land = {}
    while True:
        ret, frame = cap.read()
        
        frame = detector.findPose(frame)
        lmList = detector.findPositions(frame)

        print(lmList[14])
        
        # print frame rate on the frame 
        ct = time.time()
        fps = int(1/(ct-pt))
        pt = ct
        cv.putText(frame, str(fps), (10,30), cv.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        cv.imshow("video",frame)
        cv.waitKey(1)


if __name__ == "__main__":
    main()


