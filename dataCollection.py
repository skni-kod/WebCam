import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import time
from helpers import FixHeight, FixWeight

class Gesture:
    def __init__(self, offset, imgSize, dataPath, maxHands, camera_id, labelsPath, modelPath):
        self.cap = cv2.VideoCapture(camera_id)
        self.detector = HandDetector(maxHands=maxHands)
        self.offset = offset
        self.imgSize = imgSize
        self.dataPath = dataPath
        self.counter = 0
        self.labelsPath = labelsPath
        self.modelPath = modelPath
        self.classifier = Classifier(self.modelPath, self.labelsPath)
        self.labels = []

    def GetLabels(self):
        labelsDict = {}
        with open(self.labelsPath) as f:
            for line in f:
                splitted = list(line.split())
                key = splitted[0]
                values = [splitted[i] for i in range(len(splitted)) if i != 0]
                values = " ".join(values)
                labelsDict[int(key)] = values
        self.labels = list(labelsDict.values())
        
    def SaveImage(self, imgWhite):
        self.counter += 1
        cv2.imwrite(f"{self.dataPath}/Image_{time.time()}.jpg", imgWhite)
        print(f'Image number {self.counter} saved!')

    def Detect(self):
        success, img = self.cap.read()
        imgPrediction = img.copy()
        hands, img = self.detector.findHands(img)
        if hands:
            try:
                hand = hands[0]
                x, y, w, h = hand['bbox']
                imgWhite = np.ones((self.imgSize, self.imgSize, 3), np.uint8) * 255
                imgCrop = img[y - self.offset:y + h + self.offset, x - self.offset:x + w + self.offset]
                aspectRatio = h / w
                if aspectRatio > 1:
                    imgWhite = FixHeight(imgWhite, self.imgSize, w, h, imgCrop)
                else:
                    imgWhite = FixWeight(imgWhite, self.imgSize, w, h, imgCrop)
                prediction, index = self.classifier.getPrediction(imgWhite, draw=False)
                print(prediction, index)
                cv2.putText(imgPrediction, self.labels[index], (x , y - 20), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 0, 255), 2)
                cv2.rectangle(imgPrediction, (x - self.offset, y - self.offset), (x + w + self.offset, y + h + self.offset), (255, 0, 255), 3)
                cv2.imshow("Hand", imgCrop)
                cv2.imshow("Hand shaped", imgWhite)
            except:
                print("Change hand position!")
        cv2.imshow("Prediction", imgPrediction)    
        key = cv2.waitKey(1)
        if key == ord("s"):
            self.SaveImage(imgWhite)
