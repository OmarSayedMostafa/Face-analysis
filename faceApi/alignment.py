from faceApi.utils import xywh_2_x1y1x2y2, convert_dlib_rect_2_x1y1x2y2
from faceApi.utils import get_face_rotation_angle, crop_rotated_box
from faceApi.utils import rotate_box, get_face_crop, apply_offsets_to_face_bounding_box
import numpy as np
import cv2
from faceApi import settings
from imutils.face_utils import shape_to_np

def align(frame, faces_boundingboxes, facial_landmarks, offsets = (0, 0), target_sizes=[(96,96)], debug=True):
    face_batches = []
    #--------------------------------------
    for _ in range(len(target_sizes)):
        face_batches.append([])
    #--------------------------------------
    for i, (boundingbox, landmarks) in enumerate(zip(faces_boundingboxes, facial_landmarks)):
        boundingboxwithOffset = apply_offsets_to_face_bounding_box(boundingbox, offsets, frame.shape)
        


        face_angle = get_face_rotation_angle(landmarks)

        if face_angle == 180 or (abs(face_angle)>=170.0 and abs(face_angle)<180.0):
            # print("[not rotated] crop number ", i, " angle = ", face_angle)
            # draw_box_with_text(frame, boundingboxwithOffset, "not rotated", color=(100,50,150))
            croped = get_face_crop(frame, boundingboxwithOffset)
            
        else:
            # print("[rotated] crop number ", i, " angle = ", face_angle)
            rotated_box_points = rotate_box(boundingboxwithOffset, angle=face_angle)
            croped = crop_rotated_box(frame, rotated_box_points, face_angle, draw=False)


        for j, size in enumerate(target_sizes):
            if size[-1]==1:
                croped = cv2.cvtColor(croped, cv2.COLOR_BGR2GRAY)
            croped = cv2.resize(croped, size[:-1])
            face_batches[j].append(croped)


        # ----------------------------------
        # if settings.Debug:
        #     cv2.imshow(str(face_angle)+"_crop"+str(i), croped)
        #----------------------------------
    
    face_batches = np.array(face_batches)

    
    return face_batches

