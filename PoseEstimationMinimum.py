import cv2
import mediapipe as mp
import time

mpDraw = mp.solutions.drawing_utils
mpPose = mp.solutions.pose
pose = mpPose.Pose(static_image_mode=True) #has some parameters but for simplicity will not touch those

# Load the video
cap = cv2.VideoCapture("Videos/V1.mp4")
previousTime = 0

while True:
    #Reading the image
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = pose.process(imgRGB) #processing the image using mediapipe
    print(result.pose_landmarks) # this shows the x, y, z and the visibility of all the landmarks it got after processsing
    
    if result.pose_landmarks:
        mpDraw.draw_landmarks(img, result.pose_landmarks, mpPose.POSE_CONNECTIONS) #this will draw the points on the landmarks and will also connect the landmarks cause of mpPose.POSE_CONNECTIONS
        for id, lm in enumerate(result.pose_landmarks.landmark): #enumerate means it will the loop count in id and landmarks in lm
            
            h, w, c = img.shape #height width and channel
            cx, cy = int(lm.x*w) ,int(lm.y*h) # we have to do this cause the positions of the landmarks are in ratios with the image size
            cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED) #This will draw a opaque circle of radius 5 on each of the landmarks  
    
    #calculating FPS and printing fps
    currentTime = time.time()
    fps = 1/(currentTime - previousTime)
    previousTime = currentTime
    cv2.putText(img, str(int(fps)), (70,50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
    cv2.imshow("Image", img)

    cv2.waitKey(1) #1 millisecond Delay


#version issue mediapipe wont work above version 0.10.9 in apple M2 (maybe in intel macs not sure)