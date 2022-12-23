import math
import cv2

def FixHeight(imgWhite, imgSize, w, h, imgCrop):
     """The function adjusts the height to always be the same as the imgSize,
      gives the image of the hand to the center and returns the fixed image"""
     k = imgSize / h
     wCal = math.ceil(k * w)
     imgResize = cv2.resize(imgCrop, (wCal, imgSize))
     wGap = math.ceil((imgSize - wCal) / 2)
     imgWhite[:, wGap:wCal + wGap] = imgResize
     return imgWhite

def FixWeight(imgWhite, imgSize, w, h, imgCrop):
     """The function adjusts the weight to always be the same as the imgSize,
      gives the image of the hand to the center and returns the fixed image"""
     k = imgSize / w
     hCal = math.ceil(k * h)
     imgResize = cv2.resize(imgCrop, (imgSize, hCal))
     hGap = math.ceil((imgSize - hCal) / 2)
     imgWhite[hGap:hCal + hGap, ] = imgResize
     return imgWhite
