import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import time

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
classifier = Classifier("Model/keras_model.h5","Model/labels.txt")

offset = 22
imgSize = 350

counter = 0

labels = ["Victory", "Muka", "Salut_wolkanski", "Call_me"]

while True:
    success, img = cap.read()
    imgOutput = img.copy()
    hands, img = detector.findHands(img)
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8)*255

        imgCrop = img[y-offset:y + h+offset, x-offset:x + w+offset]
        imgCropShape = imgCrop.shape

        aspectRatio = h/w

        if aspectRatio > 1:
            k = imgSize / h
            wCal = math.ceil(k*w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            imgResizeShape = imgResize.shape
            wGap = math.ceil((350 - wCal)/2)
            imgWhite[:,wGap:wCal + wGap] = imgResize
            prediction, index = classifier.getPrediction(imgWhite, draw=False)
            print(prediction, index)

        else:
            k = imgSize / w
            hCal = math.ceil(k*h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            imgResizeShape = imgResize.shape
            hGap = math.ceil((350 - hCal)/2)
            imgWhite[hGap:hCal + hGap, :] = imgResize
            prediction, index = classifier.getPrediction(imgWhite, draw=False)
            print(prediction, index)
        
        if labels[index] == "Salut_wolkanski":
            cv2.rectangle(imgOutput, (x-offset,y-offset-35), (x-offset+200, y-offset),(255,0,255), cv2.FILLED)    
            cv2.putText(imgOutput, labels[index], (x-20,y-35),cv2.FONT_HERSHEY_COMPLEX, 0.7, (255,255,255), 1)
            cv2.rectangle(imgOutput, (x-offset,y-offset), (x+w+offset, y+h+offset),(255,0,255), 4)
        else:
            cv2.rectangle(imgOutput, (x-offset,y-offset-45), (x-offset+175, y-offset),(255,0,255), cv2.FILLED)    
            cv2.putText(imgOutput, labels[index], (x,y-35),cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,255), 2)
            cv2.rectangle(imgOutput, (x-offset,y-offset), (x+w+offset, y+h+offset),(255,0,255), 4)

        cv2.imshow("ImageCrop", imgCrop)
        cv2.imshow("ImageWhite", imgWhite)
    cv2.imshow("Image", imgOutput)
    key = cv2.waitKey(1)
    