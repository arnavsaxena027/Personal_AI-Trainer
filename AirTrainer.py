import cv2
import numpy as np
import mediapipe as mp
import time
import PoseModule as pm
    
cap = cv2.VideoCapture("Videos/V2.mp4")
#cap = cv2.VideoCapture(1)

detector = pm.poseDetector()
count = 0
dir = 0
pTime = 0

while True:
    success, img = cap.read() #for the video in the folder
    #img = cv2.resize(img, 1280, 720) #change the size of the window that opens for video when we run the code

    # img = cv2.imread("Videos/I2.jpg") # for the image in the folder
    img = detector.findPose(img, False)
    lmList = detector.findPosition(img, False)
    #print(lmList)

    if lmList != 0:
        # #RIGHT ARM
        # detector.findAngle(img, 12, 14, 16, True)

        #LEFT ARM
        angle = detector.findAngle(img, 11, 13, 15, True)
        per = np.interp(angle , (10, 160), (100, 0))
        bar = np.interp(angle, (10, 160), (200, 1500)) # 650 is minimum and 100 is maximum cause opencv convention is reverse
        color = (255, 0, 0)
        #check for the dumbbell curls
        if per == 0:
            color = (255, 255, 255)
            if dir == 0:
                count +=0.5
                dir = 1
        
        if per == 100:
            color = (255, 255, 255)
            if dir == 1:
                count+= 0.5
                dir = 0
        
        #for the bar of progress
        cv2.rectangle(img, (40, 200), (170, 1500), (0, 0, 0), 10)
        cv2.rectangle(img, (40, int(bar)), (170, 1490), color, cv2.FILLED)
        cv2.putText(img, f'{int(per)} %', (55,150), cv2.FONT_HERSHEY_PLAIN, 4,color, 10)

        #For showing the curl count
        cv2.rectangle(img, (0, 1650), (250, 1920), (0, 255, 0), cv2.FILLED) #background for count
        cv2.putText(img, str(int(count)), (55,1850), cv2.FONT_HERSHEY_PLAIN, 15, (255, 0, 0), 15) #for showing Count

    #Calculating the FPS
    cTime = time.time()
    fps = 1/(cTime- pTime)
    pTime = cTime
    #cv2.putText(img, str(int(fps)), (55,150), cv2.FONT_HERSHEY_PLAIN, 5, (255, 0, 0), 10) #Showing FPS

    

    cv2.imshow("Training Video", img)
    cv2.waitKey(1)