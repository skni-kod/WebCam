from dataCollection import Gesture

if __name__ == '__main__':
    gesture = Gesture(
        offset=20, imgSize=300, dataPath="Data/Salut_wolkanski",
        camera_id=0, maxHands=1, labelsPath="Model/labels.txt", modelPath="Model/keras_model.h5")
    gesture.GetLabels()
    while True:
        gesture.Detect()