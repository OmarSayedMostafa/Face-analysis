import cv2
import dlib
import numpy as np
from faceApi import box_utils
import onnx
import onnxruntime as ort
from onnx_tf.backend import prepare
import time 

from faceApi.static_pathes import ultra_light_640_onnx_model_path, ultra_light_320_onnx_model_path, dlib_5_face_landmarks_path
from faceApi import settings

class UltraLightOnnxFaceDetector:
    '''
    Ultra-light face detector by Linzaer and MobileFaceNet¹.
    Explained by the author at: https://towardsdatascience.com/real-time-face-recognition-with-cpu-983d35cc3ec5
    Original Repo at GitHub: https://github.com/fyr91/face_detection.git
    '''

    def __init__(self, target_model=640):
        '''
        load model weights and allocate to memory
            parameters:
                target_model: there are 2 models, one trained on image shape = [width=320, height=240]
                                and the other model trained on image shape = [width=640, height=480]
                                so you need which one you want to use.
        '''

        self.target_model=target_model
        #---------------------------------------------------------
        if self.target_model==640:
            self.onnx_model_path = ultra_light_640_onnx_model_path
        else:
            self.onnx_model_path = ultra_light_320_onnx_model_path
        #---------------------------------------------------------
        # loading target onnx model 640x480 or 320x240
        self.onnx_model = onnx.load(self.onnx_model_path)
        self.predictor = prepare(self.onnx_model)
        self.ort_session = ort.InferenceSession(self.onnx_model_path)
        self.input_name = self.ort_session.get_inputs()[0].name
        #---------------------------------------------------------
        
        self.img_mean = np.array([127, 127, 127])

    def detect(self,rgb_frame, threshold=0.6):
        '''
        detect faces from given RGB frame(opencv/numpy)
            parameters:
                rgb_frames : RGB opencv-numpy image/frame
                threshold : (float between 0.0 and 1.0) threshold value to filter detection with confidence less than the given threshold
            return:
                boxes: (numpy array) contains detected faces bounding boxes in format (x1, y1, x2, y2)/(left, top, right, bottom)
                        where (x1, y1) is the upper left corner of the box,
                                        (x2, y2) is the lower right corner of the box.  
                probs: (1d numpy array ) contains probability/confidences of detected boxes 
                width : (1d numpy array ) contains the width of each box
        '''
        h, w, _ = rgb_frame.shape
        # preprocess img acquired
        if self.target_model==640:
            img = cv2.resize(rgb_frame, (640, 480)) # resize
        else:
            img = cv2.resize(rgb_frame, (320, 240)) # resize

        img = (img - self.img_mean) / 128
        img = np.transpose(img, [2, 0, 1])
        img = np.expand_dims(img, axis=0)
        img = img.astype(np.float32)

        # predict boxes and confidences of them 
        # confidences shape (batch, 17640, 2)
        # boxes shape (batch, 17640, 4)

        confidences, boxes = self.ort_session.run(None, {self.input_name: img})

        print("\n\nconfidences shape", confidences.shape)
        print("boxes shape", boxes.shape)

        # post porcessing 
        boxes, labels, probs, widths = box_utils.predict(w, h, confidences, boxes, prob_threshold=threshold)
        
        return boxes, probs, widths

