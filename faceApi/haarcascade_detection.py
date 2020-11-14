import cv2
from faceApi.static_pathes import haarcascade_frontalface_Path


class CascadeFrontalFaceDetector:
    '''
    Haar Cascade based Face Detector
    #Pros
        1. Works almost real-time on CPU
        2. Simple Architecture
        3. Detects faces at different scales
    #Cons
        1. The major drawback of this method is that it gives a lot of False predictions.
        2. Doesn’t work on non-frontal images.
        3. Doesn’t work under occlusion
    source : https://www.learnopencv.com/face-detection-opencv-dlib-and-deep-learning-c-python/
    code on github: https://github.com/spmallick/learnopencv/tree/master/FaceDetectionComparison
    '''

    def __init__(self):
        
        self.faceCascade = cv2.CascadeClassifier(haarcascade_frontalface_Path)

    def detect(self, frame):
        # Detect faces
        # faces = self.faceCascade.detectMultiScale(frame,scaleFactor=1.1,minNeighbors=5, flags=cv2.CASCADE_SCALE_IMAGE)
        # return [(x, y, x+w, y+h) for (x,y,w,h) in faces]

        return self.faceCascade.detectMultiScale(frame,scaleFactor=1.1,minNeighbors=5, flags=cv2.CASCADE_SCALE_IMAGE)


    def convert_detection_2_x1y1x2y2(self, cascade_detections):
        return [(x, y, x+w, y+h) for (x,y,w,h) in cascade_detections]