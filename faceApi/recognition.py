import numpy as np
import multiprocessing
import face_recognition
import os
import pickle
from faceApi import settings

import cv2

class Recognizer:
    '''
    face recognition using face_recognition api
    '''
    def __init__(self, known_faces_encodes_npy_file_path, known_faces_ids_npy_file_path, face_tag_model_path=None):
        '''
        parameters:
            known_faces_encodes_npy_file_path : path to the numpy file which contains saved faces encodes 
            known_faces_ids_npy_file_path : path to the numpy file which contains corresponding faces encodes ids.
            face_tag_model_path : path of pickled MLP model to predict face gender and race from predicted face encodings
        '''
        self.known_faces_encodes_npy_file_path = known_faces_encodes_npy_file_path
        self.known_faces_ids_npy_file_path  = known_faces_ids_npy_file_path
        self.load_known_faces_encodes_ids() # load the pre recognized faces if exist.

        if face_tag_model_path !=None:
            self.faceTagModel, self.faceTagsLabels = self.load_faceTag_model(face_tag_model_path)


    def load_known_faces_encodes_ids(self):
        '''
        load pre saved faces encodes vectors and corrospondings ids from numpy files into 
        **known face encodes** list of numpy vectors that hold face encodes and **known_face_ids**(corresponding face encodes ids in list format)
        compute **max_id** to use to append new encodes with new id.
        '''
        self.known_face_encodings =[]
        self.known_face_ids = []
        self.max_id = -1
        try:
            self.known_face_encodings = np.load(self.known_faces_encodes_npy_file_path).tolist()
            self.known_face_ids = np.load(self.known_faces_ids_npy_file_path).tolist()
            self.max_id = max(self.known_face_ids)

            print("\n\n\n\ \t\t\t\t\t#### loaded ##### \n")
            print("\n[INFO] known_face_ids loaded : ", self.known_face_ids)
            print("\n[INFO] max id loaded : ", self.max_id)
            print("\t\t\t###############################\n\n")
        except Exception as e:

            print("\n\n\n\t\t\t\t #### WARNING ##### \n\n no saved file for known faces")
            os.system("touch "+self.known_faces_encodes_npy_file_path)
            os.system("touch "+self.known_faces_ids_npy_file_path)

            print(e)



    def save_faces_encodings_with_ids(self):
        '''
        save faces encodes numpy list with corresponding ids
        '''

        known_faces_encodes_np = np.array(self.known_face_encodings)
        known_faces_id_np = np.array(self.known_face_ids)

        try:
            np.save(self.known_faces_encodes_npy_file_path, known_faces_encodes_np)
            np.save(self.known_faces_ids_npy_file_path, known_faces_id_np)
        
        except Exception as e:
            print('\n\n save_faces_encodings_with_ids function in recognition cannot save ,\n', e)


    def add_new_face_encodes(self, face_encoding, save_refresh_rate=5):
        
        '''
        add new face encodes and id to the know faces encodes and ids.
        new ids will be add as (the last id added / max_id)+1
        and the max_id will be updated by max_id=max_id+1
            parameters:
                face_encoding : new face encoding to append to the known face encodes.
                save_refresh_rate : save the known faces and its ids every save_refresh_rate in the filesystem

        '''
        self.known_face_encodings.append(face_encoding)
        if len(self.known_face_ids)==0:  # if there no any ids of face encodes added yet, add 0 as the first id
            self.known_face_ids.append(0)
            face_id = 0
            self.max_id=0
        else:
            face_id = self.max_id+1
            self.max_id +=1
            self.known_face_ids.append(face_id)
        
        # update and save known faces every (save_refresh_rate) new faces
        if self.max_id%save_refresh_rate==0:
            save_process = multiprocessing.Process(target=self.save_faces_encodings_with_ids)
            save_process.start()


        return face_id



    def recognize(self, frame, faces_bounding_boxes, recognition_threshold_distance = 0.4, save_rate=1, with_gender_race=True):
        '''
        try to recognize faces in given images
            parameters:
                frame : opencv/numpy rgb frame/image
                faces_bounding_boxes : detected faces bounding boxes in format (x1, y1, x2, y2)/(left, top, right, bottom)
                                        where (x1, y1) is the upper left corner of the box,(x2, y2) is the lower right corner of the box.
                recognition_threshold_distance : verfy 2 faces if the distance between them less than or equal the given threshold
                save_rate : the number of faces encodes that the known faces will be saved to the disk after added them to the known faces list
                with_gender_race : boolen flag to feed the face encodes to face tags MLP to predict the gender and race of face
            returns:
                detected_faces_ids : list contains the recognized faces ids 
        '''

        detected_faces_ids = []

        if len(faces_bounding_boxes)> 0:
            face_locations = [(top, right, bottom, left) for [left, top, right, bottom] in faces_bounding_boxes]
            # get 1D list contains 128 encodes for each face founded in given frame 
            face_encodings, raw_landmarks = face_recognition.face_encodings(frame, face_locations)            

            # if settings.Debug:
            #     print("\nfaceencodes and land marks\n", (face_encodings, raw_landmarks))
            # get face tags
            if with_gender_race:
                faces_tags = self.faceTagModel.predict_proba(face_encodings)
            
            else:
                faces_tags= [None for _ in range(len(face_encodings))]

            # if settings.Debug:
            #     print("\nface tags\n", faces_tags)
            
            # container to contain the result of trying to recognize each face
            #loop over each face encodes found
            genders_list, races_list = [],[]
            for face_encoding, tag in zip(face_encodings, faces_tags):

                #get tags
                try:
                    gender, race = self.get_tags(tag)
                    genders_list.append(gender)
                    races_list.append(race)
                except:
                    pass

                # See if the face is  matching any of known face(s)
                # use the known face with the smallest distance to the new face
                if len(self.known_face_encodings)==0:
                    face_id = self.add_new_face_encodes(face_encoding, save_refresh_rate=save_rate)

                else:
                    face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                    best_match_index, min_distance = None, None

                    if len(face_distances) > 0: #if not that there is no any known face yet
                        best_match_index = np.argmin(face_distances)
                        min_distance = np.min(face_distances)
                        # check if the best match is less than or equal the given threshold
                        if min_distance <= recognition_threshold_distance:
                            # get the corrosponding id of the matched face
                            face_id = self.known_face_ids[best_match_index]
                            print("\n\ntesst", face_id)
                            # if the distance is very close to the threshold 
                            # then add the face encodes to avoid miss recognise it later
                            if(recognition_threshold_distance-min_distance <= recognition_threshold_distance/2.0):
                                self.known_face_encodings.append(face_encoding)
                                self.known_face_ids.append(face_id)
                        else:
                            # if the distance is not less than or equal the given threshold
                            face_id = self.add_new_face_encodes(face_encoding, save_refresh_rate=save_rate)
                    

                if face_id!=None:
                    detected_faces_ids.append(face_id)
                
        return detected_faces_ids, raw_landmarks, face_encodings, genders_list, races_list



    def get_tags(self,face_tags_probs):
        '''
        get gender and race based on predicted probabily of MLP tags model
            parameters : 
                face_tags_probs : (numpy array) contains the prediction probabilty
            returns :
                gender, race 
        '''
        gender = "Male" if face_tags_probs[0]>=0.5 else "Female"
        race = self.faceTagsLabels[1:4][np.argmax(face_tags_probs[1:4])]
        return gender, race

    def load_faceTag_model(self, face_tag_model_path):
        """
        load the pickeled face tags model
            prameters:
                face_tag_model_path : (string) contains the path of pickled model
            returns:
                clf : MLP model
                labels: 70 labels the model can predict
        """
        with open(face_tag_model_path, 'rb') as f:
            clf, labels = pickle.load(f, encoding='latin1')
        
        return clf, labels