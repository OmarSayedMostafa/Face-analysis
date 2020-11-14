from statistics import mode
import cv2
from tensorflow.keras.models import load_model
import numpy as np
from faceApi.utils import preprocess_input
from faceApi.static_pathes import emotion_model_path, gender_model_path



class EmotionExtractor:
    def __init__(self, emotion_model_path, labels_list):
        self.emotion_model_path = emotion_model_path
        self.labels_list = labels_list
        self.model = load_model(self.emotion_model_path, compile=False)
        self.input_size = self.model.input_shape[1:]

    def get_input_shape(self):
        return self.input_size

    def predict(self, faces_batch):
        '''
        extract human face emotion from given grayscale Faces Batch
        '''
        faces_batch = preprocess_input(faces_batch)
        emotion_prediction = self.model.predict(faces_batch)
        emotion_label_indecies = np.argmax(emotion_prediction, axis=-1)
        emotion_predicted_text = []
        for label_index in emotion_label_indecies:
            emotion_predicted_text.append(self.labels_list[label_index])
        return emotion_predicted_text



