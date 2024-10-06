import cv2
import mediapipe as mp
import time
import math

class poseDetector():

    def __init__(self, static_image_mode=False, model_complexity=1, smooth_landmarks=True, enable_segmentation=False, smooth_segmentation=True, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        
        self.static_image_mode = static_image_mode
        self.model_complexity = model_complexity
        self.smooth_landmarks = smooth_landmarks
        self.enable_segmentation = enable_segmentation
        self.smooth_segmentation = smooth_segmentation
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
    
        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose

        self.pose = self.mpPose.Pose(self.static_image_mode, self.model_complexity, self.smooth_landmarks, self.enable_segmentation, self.smooth_segmentation, self.min_detection_confidence, self.min_tracking_confidence) #has some parameters but for simplicity will not touch those
        
    def findPose(self, img, draw = True):
    
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.result = self.pose.process(imgRGB) #processing the image using mediapipe
        
        #print(self.result.pose_landmarks) # this shows the x, y, z and the visibility of all the landmarks it got after processsing
        if self.result.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, self.result.pose_landmarks, self.mpPose.POSE_CONNECTIONS) #this will draw the points on the landmarks and will also connect the landmarks cause of mpPose.POSE_CONNECTIONS
        
        return img
    
    def findPosition(self, img, draw = True):
        self.lmList = []
        if self.result.pose_landmarks:
            for id, lm in enumerate(self.result.pose_landmarks.landmark): #enumerate means it will the loop count in id and landmarks in lm
                
                h, w, c = img.shape #height width and channel
                #print(id, lm)
                
                cx, cy = int(lm.x*w) ,int(lm.y*h) # we have to do this cause the positions of the landmarks are in ratios with the image size
                self.lmList.append([id, cx, cy])
                
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED) #This will draw a opaque circle of radius 5 on each of the landmarks
        return self.lmList

    def findAngle(self, img, p1 , p2 , p3, draw = True):

        #get the landmarks
        _, x1, y1 = self.lmList[p1]
        _, x2, y2 = self.lmList[p2]
        _, x3, y3 = self.lmList[p3]

        #Calculate the angle
        angle = math.degrees(math.atan2(y1-y2, x1 - x2) - math.atan2(y3-y2,x3-x3))
        
        if angle <0:
            angle +=360
        
        # if angle > 180:
        #     angle = 360 - angle

        # print(angle)


        #Draw 
        if draw:
            cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), 10)
            cv2.line(img, (x3, y3), (x2, y2), (255, 255, 255), 10)
            cv2.circle(img, (x1, y1), 35, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x1, y1), 50, (0, 0, 255), 2)

            cv2.circle(img, (x2, y2), 35, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), 50, (0, 0, 255), 2)
            
            cv2.circle(img, (x3, y3), 35, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x3, y3), 50, (0, 0, 255), 2)
            cv2.putText(img, str(int(angle)), (x2 + 30, y2 + 25), cv2.FONT_HERSHEY_PLAIN, 5, (255, 0, 255), 5) #Last number is for thickness and first is for scale

        return angle


def main():
    # Load the video
    cap = cv2.VideoCapture("Videos/Video2.mp4")
    previousTime = 0

    detector = poseDetector()
    while True:
        #Reading the image
        success, img = cap.read()
        img = detector.findPose(img)
        lmList = detector.findPosition(img)
        print(lmList)

        # #CAN ALSO DO THIS WHERE WE ONLY DRAW THE CIRCLES FOR A SPECIFIC POINT AND NOT ALL
        # if len(lmList != 0):
            # lmList = detector.findPosition(img, draw = False)
            # cv2.cricle(img, lmList[14][1], lmList[14][2], 5, (255, 0, 0), cv2.FILLED)

        #calculating FPS and printing fps
        currentTime = time.time()
        fps = 1/(currentTime - previousTime)
        previousTime = currentTime
        cv2.putText(img, str(int(fps)), (70,50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
        cv2.imshow("Image", img)

        cv2.waitKey(1) #1 millisecond Delay

if __name__ == "__main__": 
    main()
