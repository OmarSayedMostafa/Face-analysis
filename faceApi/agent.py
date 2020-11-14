import numpy as np
import cv2, os

from faceApi.alignment import align
from faceApi import settings
from faceApi.recognition import Recognizer
from faceApi.emotions import EmotionExtractor
from faceApi.utils import get_labels
from faceApi.static_pathes import known_faces_encodes_npy_file_path, known_faces_ids_npy_file_path
from faceApi.static_pathes import emotion_model_path, faceTageModelPath
from faceApi.ultra_light_Detector import UltraLightOnnxFaceDetector
from face_recognition.api import _raw_face_landmarks

# initialize the face detection model
ultraLightFD = UltraLightOnnxFaceDetector(target_model=640)# target_model = 320*240/640*480
print("\n[INFO]loaded face detector Model successfully!")
# Initialize face recognition class
recognizer = Recognizer(known_faces_encodes_npy_file_path, known_faces_ids_npy_file_path, face_tag_model_path=faceTageModelPath)
print("\n[INFO]loaded face recognizer Model successfully!")
# emotion extractor 
emotions_labels = get_labels('fer2013') # fer2013 : is the dataset name
emotion_extractor = EmotionExtractor(emotion_model_path,emotions_labels)
print("\n[INFO]loaded emotion_extractor Model successfully!")

#-----------------------------------------------------------------------------------

def face_detection(frame, detection_threshold = 0.9):
    '''
    detected faces in given frame using flobal model ultraLightFD
        parameters : 
            frame :               (numpy 2d array)            rgb opencv / numpy fram/image.
            
            detection_threshold : (float between 0.0 and 1.0) detection threshold that any detected face with less confidence than threshold will be ignored.
        
        returns:
            faces_locations: (numpy array) contains each detected faces bounding boxes in format (x1, y1, x2, y2) or (left, top, right, bottom), where (x1, y1) is the upper left corner of the box, (x2, y2) is the lower right corner of the box.  
            faces_confidences: (1d numpy array of floar between 0.0 and 1.0) contains probability/confidences of detected boxes 
            width : (1d numpy array of int) contains the width of each detected face.
    
    '''
    # tensorflow model in ONNX framework to detect faces
    faces_locations, faces_confidences, widths =  ultraLightFD.detect(frame, threshold=detection_threshold)
    return faces_locations, faces_confidences, widths

# -----------------------------------------------------------------------------------

def select_topK_faces(faces_container,dicarded_faces_container, k=10):
    '''
    sort faces by its confidences and select the top K faces of them
        parameters :
            faces_container :           (list of list) contains each face [location, confidence, width].
            
            dicarded_faces_container :  (list of list) contains the discarded faces location to append to it the remains faces that does not in top selected k.
            
            k : the number of top faces to be selected.
        
        returns : 
            selected_faces_locations :    (list of [x1,y1,x2,y2])                 the top k selected faces locations
            selected_faces_confidences :  (list of floats between 0.0 and 1.0)    the top k selected faces confidences
            selected_faces_widths :       (list of int)                           the top k selected faces widths
            dicarded_faces_container :    (list of [x1,y1,x2,y2])                 list of the remaining faces that haven't been selected with the top k faces.

    '''
    # sort faces by confidences
    confidence_index = 1
    width_index = 2
    # sort the faces by confidence and if 2 is equal in confidence sort them by width.
    sorted_faces  = sorted(faces_container, key=lambda x: (x[confidence_index],x[width_index]), reverse=True)
    # select top k faces or all faces if faces count is less than top k required
    top_picked_faces_count = min(len(sorted_faces), k)
    # initailize the return variables
    selected_faces_locations, selected_faces_confidences, selected_faces_widths = [], [], []
    for t in range(top_picked_faces_count):
        selected_faces_locations.append(sorted_faces[t][0])
        selected_faces_confidences.append(sorted_faces[t][1])
        selected_faces_widths.append(sorted_faces[t][2])
    
    # add the remaining faces to the discarded list.
    for j in range(top_picked_faces_count,len(faces_container)):
        dicarded_faces_container.append((sorted_faces[j][0], sorted_faces[j][1], sorted_faces[j][2]))

    return selected_faces_locations, selected_faces_confidences, selected_faces_widths, dicarded_faces_container

# -------------------------------------------------------------------------------------

def filter_faces_with_minimum_width(faces_locations, faces_confidences, faces_widths, minimum_face_width=48):
    '''
    filter list of faces by its width
        parameters:
            faces_locations: (numpy array/list) contains each detected faces bounding boxes in format (x1, y1, x2, y2) or (left, top, right, bottom), where (x1, y1) is the upper left corner of the box, (x2, y2) is the lower right corner of the box.  
            
            faces_confidences: (1d numpy array of floar between 0.0 and 1.0) contains probability/confidences of detected boxes 
            
            width : (1d numpy array of int) contains the width of each detected face.

            minimum_face_width : minimum face width which discard anyface with width less than minimum.

        returns:
            filtered_faces : (list of [face_location, confidenc, width]) of filted faces
            dicarded_faces : (list of [face_location, confidenc, width]) of discarded faces.
    '''
    filtered_faces = [(face_loc, confidence, width) for (face_loc, confidence, width) in zip(faces_locations, faces_confidences, faces_widths) if width>= minimum_face_width]
    dicarded_faces = [(face_loc, confidence, width) for (face_loc, confidence, width) in zip(faces_locations, faces_confidences, faces_widths) if width< minimum_face_width]

    return filtered_faces ,dicarded_faces

# -------------------------------------------------------------------------------------

def get_faces_landmarks(frame, faces_locations):
    '''
    get landmarks for each detected face
        parameters :
            frame : (numpy 2d array)            rgb opencv / numpy fram/image.
            faces_locations: (numpy array/list) contains each detected faces bounding boxes in format (x1, y1, x2, y2) or (left, top, right, bottom), where (x1, y1) is the upper left corner of the box, (x2, y2) is the lower right corner of the box. 
        
        returns :
            faces_landmarks : (list of dlib full_object_detection object) that contains the facial landmarks for each face.

    '''
    faces_landmarks = _raw_face_landmarks(frame, faces_locations, model="small") # face_recogintion.api lib
    return faces_landmarks

# -------------------------------------------------------------------------------------

def get_faces_batches(frame, faces_locations, faces_landmarks, models_target_sizes=[]):
    '''
    get faces batches with different size for different models.
        parameters :
            frame :                 (numpy 2d array)   rgb opencv / numpy fram/image.
            
            faces_locations:        (numpy array/list) contains each detected faces bounding boxes in format (x1, y1, x2, y2) or (left, top, right, bottom), where (x1, y1) is the upper left corner of the box, (x2, y2) is the lower right corner of the box. 

            faces_landmarks :       (list of dlib full_object_detection object) that contains the facial landmarks for each face.

            models_target_sizes :   (list of 2d tuple ex: [(width, height), (...),... ]) : to return batches for each model with its suitable input shape
        
        returns : 
            model_faces_batches :   (list of numpy arrays as batches) list of batches each batch contains faces with same shape [[batch_size, w, h, channels]]
    '''
    model_faces_batches = align(frame, faces_locations, faces_landmarks,target_sizes=models_target_sizes)
    return model_faces_batches

# --------------------------------------------------------------------------------------

def get_face_ids(frame,faces_locations, recognition_threshold_distance = 0.4, save_rate=5, with_gender_race = False):
    '''
    try to recognize given faces and assign new id for those who r not known or seen before.
        parameters : 
            frame : (numpy 2d array) rgb opencv / numpy fram/image.
            
            recognition_threshold_distance : (float between 0.0 and 1.0) verfy 2 faces if the distance between them less than or equal the given threshold, the allowed distance between known face encodes and detected one , the lower the more accurate.
            
            save_rate : (int) the number of new faces encodes with its new ids that will be saved to the disk after added them to the known faces list.
                        
            with_gender_race :  (boolen) true if want to use face encodings that predicted by recoginition net to feed to trained MLP to classify the gender and race.


        returns:

            detected_faces_ids :       (list of int) contains the ids of recognized faces if the recognize paramter flag is True, otherwise will return an empty list because there is no recognition process.
            
            faces_landmarks :          (list of dlib full_object_detection object) that contains the facial land marks for each face
            
            faces_encodes :            (list of numpy array with shape of (128,)) contains each face estimated encodes. 
            
            genders :                  (list of strings) contains the gender(Male/Female) of each face.
            
            races :                    (list of strings) contains the race (white, black, asian, etc..) of each face.

    '''
    detected_faces_ids, faces_landmarks, faces_encodes, genders, races = recognizer.recognize(frame, faces_locations, recognition_threshold_distance=recognition_threshold_distance, save_rate=save_rate, with_gender_race=with_gender_race)
    return detected_faces_ids, faces_landmarks, faces_encodes, genders, races

# --------------------------------------------------------------------------------------

def face_analysis(frame, detection_threshold = 0.6 , recognition_threshold_distance = 0.4, save_new_ids_rate=5, 
                    minimum_face_width=40, top_picked_faces=10, recognize = True,
                    recogniion_with_race_gender=True, anaylize_emotions=False):

    '''
    face detection and recognition with gender, race and emotions detection
        parameters : 
            frame :                             (numpy 2d array)            rgb opencv / numpy fram/image.
            
            detection_threshold :               (float between 0.0 and 1.0) detection threshold that any detected face with less confidence than threshold will be ignored.
            
            recognition_threshold_distance :    (float between 0.0 and 1.0) verfy 2 faces if the distance between them less than or equal the given threshold,
                                            the allowed distance between known face encodes and detected one , the lower the more accurate.
            
            save_new_ids_rate :                 (int) the number of new faces encodes with its new ids that will be saved to the disk after added them to the known faces list.
            
            minimum_face_width :                (int) the minmum width of acceptable detected face, ignore any detected face with width less than the minimum.
            
            top_picked_faces :                  (int) sort faces by confidence and select only the top K faces to work with.
            
            recognize :                         (boolen) true if face recogonition process is required, false other wise
            
            recogniion_with_race_gender :       (boolen) true if want to use face encodings that predicted by recoginition net to feed to trained MLP to classify the gender and race. (NOTE: recognize parameter should be true to use the predicted encodes to feed to the MLP(facetag model) other wise this parameter will be ignored.

            anaylize_emotions :                 (boolen) true if u want to analyize faces emotions.


        returns:
            
            selected_faces_locations : (list of list contains 4 elements (x1, y1, x2, y2)), ex:[[x1,y1,x2,y2], [...], ...] the faces location in given frame after been filterd by minimum width and selected the top k faces of all faces.

            faces_confidences :        (list of floats between 0.0 and 1.0) contains the confidence of prediction of each face.

            detected_faces_ids :       (list of int) contains the ids of recognized faces if the recognize paramter flag is True, otherwise will return an empty list because there is no recognition process.
            
            faces_landmarks :          (list of dlib full_object_detection object) that contains the facial land marks for each face

            genders :                  (list of strings) contains the gender(Male/Female) of each face.
            races :                    (list of strings) contains the race (white, black, asian, etc..) of each face.
            emotions :                 (list of strings) contains the emotion (happy, angry, sad, fear, etc..) of each face.
            dicarded_faces :           (list of list contains 4 elements (x1, y1, x2, y2)), ex:[[x1,y1,x2,y2], [...], ...] the discarded faces on the two filtering proces(minimum width and top k selected).
    '''

    # initialize all returned data to be empty.
    selected_faces_locations,faces_confidences, detected_faces_ids, emotions = [], [], [], []
    faces_landmarks, genders, races, dicarded_faces = [], [], [], []
    

    if settings.Debug:
        print("\n[INFO][face_analysis_agent] recived frame with shape = ", frame.shape)
    
    # step one detect faces
    faces_locations, faces_confidences, faces_widths = face_detection(frame, detection_threshold=detection_threshold)

    # check if no faces found in the given frame return empty lists, mean nothing found.
    if len(faces_locations)==0:
        return -1
    

    # filter the founded faces with minimum width (filter phase 1)
    filtered_faces, dicarded_faces = filter_faces_with_minimum_width(faces_locations, faces_confidences, faces_widths, minimum_face_width=minimum_face_width)

    # if there are remaining faces after width filtering process (continue to filter phase 2)
    if len(filtered_faces) > 0:
        # choose K top faces from all detected faces (filter phase 2)
        selected_faces_locations, faces_confidences, _, dicarded_faces = select_topK_faces(filtered_faces, dicarded_faces, k=top_picked_faces)
        
        # check if recogonize process required.
        if recognize:
            # will return the genders and races if with_gender_race flage is equal to true.
            detected_faces_ids, faces_landmarks, _, genders, races = get_face_ids(frame,selected_faces_locations, recognition_threshold_distance = recognition_threshold_distance, save_rate=save_new_ids_rate, with_gender_race = recogniion_with_race_gender)


        # if emotion anaylise required
        if anaylize_emotions:

            if len(faces_landmarks)==0: #if faces land marks not detectd before, detect them now!
                # reformating the faces bounding box to fit dlib format 
                face_locations_for_landMarks = [(top, right, bottom, left) for [left, top, right, bottom] in selected_faces_locations]
                # use face_recognition api to get facial landmarks (5 points model)
                faces_landmarks = _raw_face_landmarks(frame, face_locations_for_landMarks, model='small')

            # get aligned face batches to feed to emotion extractor model.
            faces_batch = align(frame, selected_faces_locations, faces_landmarks,target_sizes=[emotion_extractor.get_input_shape()])

            emotions_batch = faces_batch[0]
            emotions = emotion_extractor.predict(emotions_batch)

    return selected_faces_locations,faces_confidences, detected_faces_ids, faces_landmarks, genders, races, emotions, dicarded_faces
