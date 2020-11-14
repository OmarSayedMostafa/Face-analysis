# from yoloApi.agent import cv2, yolov4

from faceApi.agent import  face_analysis, cv2
from faceApi.drawing import draw_discard_faces, draw_accepted_faces, draw_raw_land_marks
import time




def main(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_width = int(640) 
    frame_height = int(480) 
   
    size = (frame_width, frame_height) 
    result = cv2.VideoWriter('result.avi',  
                         cv2.VideoWriter_fourcc(*'MJPG'), 
                         5, size) 
    counter = 0
    while True:
        counter +=1
        try:
            # print(counter)
            if counter % 5 == 0: 
                ret, frame_read = cap.read() 
                if(ret==False):
                    print('unable to read further more from feed!')
                    break
                start_time = time.time()
                frame_rgb = frame_read

                # yolo_detections = yolov4.detect_image(frame_rgb, detection_threshold=0.40)
                # frame_rgb = yolov4.cvDrawBoxes(yolo_detections, frame_rgb)
                try:
                    faces_info = face_analysis(frame_rgb, detection_threshold=0.88, recognize=True, minimum_face_width=50, top_picked_faces=10, recogniion_with_race_gender=True, anaylize_emotions=True)

                    if faces_info == -1:
                        continue
                except Exception as e:
                    print("\n\nException : ", e)
                    continue
                
                selected_faces_locations,faces_confidences, detected_faces_ids, faces_landmarks, genders, races, emotions, dicarded_faces = faces_info


                draw_discard_faces(frame_rgb, dicarded_faces)
                draw_accepted_faces(frame_rgb, selected_faces_locations,faces_confidences, detected_faces_ids, faces_landmarks, genders, races, emotions)

                draw_raw_land_marks(frame_rgb, faces_landmarks)
                frame_rgb = cv2.resize(frame_rgb, size)
                result.write(frame_rgb) 
                
                cv2.imshow("frame", frame_rgb)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            
                print('frame processed in ', time.time()-start_time, 'Sec')
            else:
                ret, frame_read = cap.read()

        except Exception as e:
            print(e)
            exit(1)

        

        

    
    cap.release()

if __name__ == "__main__":
    main('./faceApi/samples_videos/conf_test.mp4')
