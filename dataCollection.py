import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import time
from helpers import fix_height, fix_weight

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
offset = 20
imgSize = 300
folder = "Data/Salut_wolkanski"
counter = 0

while True:
    success, img = cap.read()
    hands, img = detector.findHands(img)
    if hands:
        try:
            hand = hands[0]
            x, y, w, h = hand['bbox']
            imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
            imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]
            aspectRatio = h / w
            if aspectRatio > 1:
                imgWhite = fix_height(imgWhite, imgSize, w, h, imgCrop)
            else:
                imgWhite = fix_weight(imgWhite, imgSize, w, h, imgCrop)
            cv2.imshow("Hand", imgCrop)
            cv2.imshow("Hand shaped", imgWhite)
        except:
            print("Change hand position!")
            continue

    cv2.imshow("Camera", img)
    key = cv2.waitKey(1)
    if key == ord("s"):
        counter += 1
        cv2.imwrite(f"{folder}/Image_{time.time()}.jpg", imgWhite)
        print(counter)
