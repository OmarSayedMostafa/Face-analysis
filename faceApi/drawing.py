import cv2


def draw_discard_faces(frame, discard_faces, color=(0,0,255), thickness=2):
    for [location, confidance, width] in discard_faces:
        x1, y1, x2, y2 = location
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        cv2.putText(frame, "conf :" + str(round(confidance*100, 2)), (x1, y1 - 30 ), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, thickness)
        cv2.putText(frame, "width : "+str(int(width)), (x1, y1 - 50 ), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, thickness)

    return frame





def draw_raw_land_marks(frame, raw_landmarks):

    for landmarks in raw_landmarks:
        # print(landmarks)
        for p in landmarks.parts(): 
            cv2.circle(frame, (p.x, p.y), 3, (150, 0, 255),thickness=2)
    
    return frame





def draw_accepted_faces(frame, faces_locations,faces_confidences, detected_faces_ids, faces_landmarks, genders, races, emotions, color=(0,255,0), thickness=2):
    
    for i, location in enumerate(faces_locations):
        j = 10
        x1, y1, x2, y2 = location
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        cv2.putText(frame, "conf :" + str(round(100*faces_confidences[i],2)), (x1, y1 - j ), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, thickness)
        j+=20

        if len(detected_faces_ids) > 0:
            try:
                cv2.putText(frame, "id : "+str(detected_faces_ids[i]), (x1, y1 - j ), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, thickness)
                j+=20
            except:
                pass

        # if len(faces_landmarks) > 0:
        #     try:

        #     except:
        #         pass

        if len(genders)>0:
            try:
                cv2.putText(frame,str(genders[i]), (x1, y1 - j ), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, thickness)
                j+=20
                cv2.putText(frame,str(races[i]), (x1, y1 - j ), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, thickness)
                j+=20

            except:
                pass

        if len(emotions)>0:
            try:
                cv2.putText(frame,str(emotions[i]), (x1, y1 - j ), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, thickness)
                j+=20

            except:
                pass

    return frame














def draw_face_boxes_with_one_text(frame, face_locations, confidances, face_widths ,text,  color, thickness=2):
    for location, confidence, width in zip(face_locations, confidances, face_widths):
        x1, y1, x2, y2 = location
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        cv2.putText(frame, text, (x1, y1 - 10 ), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, thickness)
        cv2.putText(frame, "conf :" + str(confidence), (x1, y1 - 30 ), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, thickness)
        cv2.putText(frame, "width : "+str(width), (x1, y1 - 50 ), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, thickness)

    return frame



def draw_analysis_prediction(frame,filterd_faces_info, color, thickness):
    
    for face_info in filterd_faces_info:
        face_loc, face_confidence, face_id, face_landmark, gender, race, emotions = face_info
        x1, y1, x2, y2 = face_loc
        # Draw a bounding box.
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

        # Display the label at the top of the bounding box
        label_size, base_line = cv2.getTextSize(face_id, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

        y1 = max(y1, label_size[1])

        frame = cv2.putText(frame, face_confidence, (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        # frame = cv2.putText(frame, face_id, (left, top - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOR_YELLOW, 1)
        # frame = cv2.putText(frame, emotion_text, (left, top - 32), cv2.FONT_HERSHEY_SIMPLEX, 0.6,COLOR_RED, 1)
        # frame = cv2.putText(frame, gender_text, (left, top - 48), cv2.FONT_HERSHEY_SIMPLEX, 0.6,COLOR_RED, 1)

    return frame





# Default colors
COLOR_BLUE = (255, 0, 0)
COLOR_GREEN = (0, 255, 0)
COLOR_RED = (0, 0, 255)
COLOR_WHITE = (255, 255, 255)
COLOR_YELLOW = (0, 255, 255)



# Draw the predicted bounding box
def draw_box_with_text(frame, box, text, color):
    '''
    draw box in image with a given text
        parameters:
            frame: opencv/numpy image
            box : box points in format (left, top, right, bottom)/(x1,y1,x2, y2)
            text : the string to write over the box
            color : opencv color tuple BGR format (B_value, G_value, R_value) each value in range(0,255)
        return :
            frame : the frame/image with the box and the text on it
    '''
    x1,y1,x2,y2 = box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)
    frame = cv2.putText(frame, text, (x1, y1 - 10 ), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
    return frame




def draw_list_of_boxes_with_text(frame, boxes, text, color):
    '''
    draw list of boxes of a frame/image
        parameters:
            frame: opencv/numpy image
            boxes : list of boxes, each box points in format (left, top, right, bottom)/(x1,y1,x2, y2)
            text : the string to write over each of the boxes
            color : opencv color tuple BGR format (B_value, G_value, R_value) each value in range(0,255)
        return :
            frame : the frame/image with the boxes and the text on each box
    '''
    for i, box in enumerate(boxes):
        frame = draw_box_with_text(frame, box, text+" "+str(i+1), color)
    
    return frame





def draw_prediction(frame, confidance, face_id, left, top, right, bottom, emotion_text, gender_text):
    # Draw a bounding box.
    cv2.rectangle(frame, (left, top), (right, bottom), COLOR_YELLOW, 2)

    # Display the label at the top of the bounding box
    label_size, base_line = cv2.getTextSize(face_id, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

    top = max(top, label_size[1])

    frame = cv2.putText(frame, confidance, (left, top - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOR_YELLOW, 1)
    frame = cv2.putText(frame, face_id, (left, top - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOR_YELLOW, 1)
    frame = cv2.putText(frame, emotion_text, (left, top - 32), cv2.FONT_HERSHEY_SIMPLEX, 0.6,COLOR_RED, 1)
    frame = cv2.putText(frame, gender_text, (left, top - 48), cv2.FONT_HERSHEY_SIMPLEX, 0.6,COLOR_RED, 1)

    return frame




def draw_rotated_box(frame, box, color, center=None):

    '''
    draw rotated box
    '''

    p1, p2, p3, p4 = box

    cv2.line(frame,p1,p2,color,thickness=2)
    cv2.line(frame,p2,p3,color,thickness=2)
    cv2.line(frame,p3,p4,color,thickness=2)
    cv2.line(frame,p4,p1,color,thickness=2)

    return frame, [p1, p2, p3, p4]



def draw_face_info(frame, face_info):
    bounding_boxes, confidences,  detected_ids, emotions, genders = face_info
    # Display the results
    for (left, top, right, bottom), confidence,  detected_id, emotion, gender  in zip(bounding_boxes, confidences,  detected_ids, emotions, genders ):
        frame = draw_prediction(frame, str(np.round(confidence, 2)), 'id='+str(detected_id), left, top, right, bottom, emotion, gender)

    return frame
