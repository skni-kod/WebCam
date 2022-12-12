import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)

offset = 20
imgSize = 300

folder = "Data/Salut_wolkanski"
counter = 0

def dostosuj_height(imgWhite, imgSize, w, h, imgCrop):
     """Funkcja dostosowuje height by byl zawsze taki sam jak imgSize i daje obraz dloni
      na srodek i zwraca poprawiony obraz"""
     k = imgSize / h
     wCal = math.ceil(k * w)
     imgResize = cv2.resize(imgCrop, (wCal, imgSize))
     wGap = math.ceil((imgSize - wCal) / 2)
     imgWhite[:, wGap:wCal + wGap] = imgResize
     return imgWhite

def dostosuj_weight(imgWhite, imgSize, w, h, imgCrop):
     """Funkcja dostosowuje weight by byl zawsze taki sam jak imgSize i daje obraz dloni 
     na srodek i zwraca poprawiony obraz"""
     k = imgSize / w
     hCal = math.ceil(k * h)
     imgResize = cv2.resize(imgCrop, (imgSize, hCal))
     hGap = math.ceil((imgSize - hCal) / 2)
     imgWhite[hGap:hCal + hGap, ] = imgResize
     return imgWhite


while True:
    """Glowna czesc programu"""
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
                imgWhite = dostosuj_height(imgWhite, imgSize, w, h, imgCrop)
            else:
               imgWhite = dostosuj_weight(imgWhite, imgSize, w, h, imgCrop)
            cv2.imshow("Dlon", imgCrop)
            cv2.imshow("Dlon dobre wymiary", imgWhite)
        except:
            print("Zmien polozenie dloni!")
            continue

    cv2.imshow("Kamera", img)
    key = cv2.waitKey(1)
    if key == ord("s"):
        counter += 1
        cv2.imwrite(f"{folder}/Image_{time.time()}.jpg", imgWhite)
        print(counter)

