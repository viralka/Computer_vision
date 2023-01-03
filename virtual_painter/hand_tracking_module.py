import cv2
import mediapipe as mp
import time

class handDetector():
    def __init__(self, mode=False, maxHands=1, modelComplexity=1, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.modelComplex = modelComplexity
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        self.mpDraw = mp.solutions.drawing_utils

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.modelComplex, 
                                        self.detectionCon, self.trackCon)
        self.tip_id = [4,8,12,16,20]

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks:
            for hand_landmarks in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, hand_landmarks, self.mpHands.HAND_CONNECTIONS)  # mpHands.HAND_CONNECTIONS joins the landmark points

        return img

    def findPosition(self, img, handNumber=0, draw=True):
        height, width, _ = img.shape
        self.landmark_list = []

        if self.results.multi_hand_landmarks:
            hand_landmarks = self.results.multi_hand_landmarks[handNumber]
            for id, landmark in enumerate(hand_landmarks.landmark):
                height, width, channel = img.shape
                x,y,z = int(landmark.x * width), int(landmark.y * height), landmark.z
                # print(id, x, y)
                self.landmark_list.append([id,x,y])
                if draw:
                    if id==4:
                        cv2.circle(img, (x,y), 10, (255,0,0), cv2.FILLED)
                    # self.mpDraw.draw_landmarks(img, hand_landmarks, mpHands.HAND_CONNECTIONS)  # mpHands.HAND_CONNECTIONS joins the landmark points

        return self.landmark_list

    def findFingersUp(self):
        fingers = []
        # Thumb
        if self.landmark_list[4][1] < self.landmark_list[3][1]:
            fingers.append(1)
        else:
            fingers.append(0)
        
        # fore fingers
        for item in self.tip_id[1:]:
            if self.landmark_list[item][2] < self.landmark_list[item-2][2]:
                fingers.append(1)
            else:
                fingers.append(0)

        return fingers

def main():
    video_capture = cv2.VideoCapture(0)

    prev_time=0
    current_time=0
    detector = handDetector()
    while True:
        _, img = video_capture.read()
        
        img = detector.findHands(img)
        landmark_list = detector.findPosition(img)
        if landmark_list:
            print(landmark_list[8])

        current_time = time.time()
        fps = 1/(current_time-prev_time)
        prev_time = current_time

        cv2.putText(img, f'FPS: {str(int(fps))}', (10,70), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0,0,255), 2)
        cv2.imshow("Image", img)
        cv2.waitKey(1)


if __name__=="__main__":
    main()