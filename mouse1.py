
import cv2
from cvzone.HandTrackingModule import HandDetector
import mouse
import numpy as np


detector = HandDetector(detectionCon=0.9, maxHands=1)



cap = cv2.VideoCapture(0)
cam_w,cam_h = 640, 480
cap.set(3, cam_w)
cap.set(4, cam_h)
while True:
    success, img = cap.read()
    img = cv2.flip(img,1)
    hands, img = detector.findHands(img, flipType=False)
    if hands:
        lmlist = hands[0]['lmList']
        ind_x, ind_y = lmlist[8][0],lmlist[8][1]
        cv2.circle(img, (ind_x, ind_y), 5, (0, 255, 255), 2)
        conv_x = int(np.interp(ind_x,(0, cam_w),(0, 1536)))
        conv_y = int(np.interp(ind_y,(0, cam_h),(0, 864))) 
        mouse.move(conv_x, conv_y)       



    cv2.imshow("Camera", img)
    key = cv2.waitKey(1)
    if key == ord("q"):
        break