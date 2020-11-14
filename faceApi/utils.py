import time
import numpy as np
import cv2, dlib
import string, random
from imutils.face_utils import shape_to_np, FACIAL_LANDMARKS_5_IDXS, FACIAL_LANDMARKS_68_IDXS
from scipy import ndimage
import math



def iseven(value):
    '''
    check if the number is evern
        parameters:
            value : (number) to be checked if even or odd
        returns:
            boolen (True or false) true if the value/number is even , false other wise
    '''
    if value%2==0:
        return True

    return False



def calculate_margin(value1, value2):
    '''
    calculate the margin between 2 values
        parameters:
            value1: (number)
            value2: (number)
        return the margin between 2 values divided by 2
    '''
    margin = abs(value1-value2)
    if not iseven(margin):
        margin+=1
    
    margin = margin//2
    return margin



def square_a_rectangle(rect, image_shape):
    '''
    make a rectangle a square 
        parameters : 
            rect : x1, y1, x2, y2 // left, top, right, bottom
            image_shape : [height, width, channels]
        return : 
            x1, y1, x2, y2 : the new square points 
    '''
    x1, y1, x2, y2 = rect
    img_width = image_shape[1]
    img_height = image_shape[0]

    width = x2-x1
    height = y2-y1

    # check if width is greater than height
    # then add margin to height
    if width > height:
        margin = calculate_margin(width, height)
        y1 = y1-margin if y1-margin >=0 else 0
        y2 = y2+margin if y2+margin < img_height else img_height-1
    
    # check if height is greater than width
    # then add margin to width
    elif height > width:
        margin = calculate_margin(height, width)
        x1 = x1-margin if x1-margin >=0 else 0
        x2 = x2+margin if x2+margin < img_width else img_width-1

    return x1, y1, x2, y2
    
    

def crop_rotated_box(img, rotated_box, angle, draw=False):
    '''
    crop rotated box from given image
        parameters :
            img : opencv/numpy image 
            rotated_box : rotated box point [bottom left, top left, top right, bottom right]
            angle : the angle of the rotation of the box
        
        return : 
            wraped : croped area of the rotated rectangle
    '''
    cnt = np.array([
            [rotated_box[0]],
            [rotated_box[1]],
            [rotated_box[2]],
            [rotated_box[3]]
        ])
    
    rect = cv2.minAreaRect(cnt)

    # the order of the box points: bottom left, top left, top right, bottom right
    box = cv2.boxPoints(rect)
    box = np.int0(box)

    # print("bounding box: {}".format(box))
    if draw:
        cv2.drawContours(img, [box], 0, (0, 100, 155), 2)

    # get width and height of the detected rectangle
    width = int(rect[1][0])
    height = int(rect[1][1])

    src_pts = box.astype("float32")
    # coordinate of the points in box points after the rectangle has been
    # straightened
    dst_pts = np.array([[0, height-1],
                        [0, 0],
                        [width-1, 0],
                        [width-1, height-1]], dtype="float32")

    # the perspective transformation matrix
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)

    # directly warp the rotated rectangle to get the straightened rectangle
    warped = cv2.warpPerspective(img, M, (width, height))

    # if the angle < 0 mean that the box is rotated anti clock wise, 
    # so the result image should be rotated by 90 degree
    warped = ndimage.rotate(warped, 90) if angle < 0 else warped

    return warped
    



def convert_box_2_dlib_rect(box):
    '''
    convert box to dlib rect
        parameter : 
            box : tuple/list contains for values : left x, top y, right x, bottom y or x1,y1,x2,y2
        return :
            dlib.rectangle(left, top, right, bottom) : dlib rectangle class with the box points
    '''
    left, top, right, bottom = box
    return dlib.rectangle(left, top, right, bottom)




def get_face_rotation_angle(landmarks):
    
    '''
    get the face rotation 
    by calculating the slop of eyes herizontal line from the landmarks provided

    parameters:
        landmarks: facial landmarks {the 5 landmarks or 68 landmarks}
    
    return :
        angle: the rotation angle of the face, negative angles mean rotation counter clockwise
    '''

    if type(landmarks) == dlib.full_object_detection:
        shape = shape_to_np(landmarks)
    else:
        shape = landmarks

    # get eyesy points
    if (len(shape) == 68):
        # extract the left and right eye (x, y)-coordinates
        (lStart, lEnd) = FACIAL_LANDMARKS_68_IDXS["left_eye"]
        (rStart, rEnd) = FACIAL_LANDMARKS_68_IDXS["right_eye"]
    else: # mean that the 5 landmarks points r used
        (lStart, lEnd) = FACIAL_LANDMARKS_5_IDXS["left_eye"]
        (rStart, rEnd) = FACIAL_LANDMARKS_5_IDXS["right_eye"]

    leftEyePts = shape[lStart:lEnd]
    rightEyePts = shape[rStart:rEnd]


    # compute the center of mass for each eye
    leftEyeCenter = leftEyePts.mean(axis=0).astype("int")
    rightEyeCenter = rightEyePts.mean(axis=0).astype("int")

    # compute the angle between the eye centroids
    dY = rightEyeCenter[1] - leftEyeCenter[1]
    dX = rightEyeCenter[0] - leftEyeCenter[0]
    angle = np.degrees(np.arctan2(dY, dX))


    return angle
    


def convert_dlib_rect_2_x1y1x2y2(dlib_rect):
    '''
    convert dlib rectangle to box points list
        parameter:
            dlib_rect : dlib.rectangle class 
        return : 
            x1, y1, x2, y2: the left,top, right, bottom of the box
    '''
    return (dlib_rect.left(), dlib_rect.top(), dlib_rect.right(), dlib_rect.bottom())

    
def xywh_2_x1y1x2y2(rect):
    '''
    convert box format from x y width height to left, top, right, bottom
        parameters : 
            rect: list or tuple contains (x, y, width, height)
        return : 
            x1, y1, x2, y2
    '''
    x, y, w, h = rect
    return x, y, x+w, y+h


    

def get_face_crop(frame, face_location):
    '''
    crop face from given image/frame
        parameters : 
            frame: opencv/numpy image
            face_location : face box coordinates in format (x1, y1, x2, y2) or (left, top, right, bottom)
    '''
    for i in range(len(face_location)):
        face_location[i] = face_location[i] if face_location[i]>=0 else 0
        
    x1, y1, x2, y2 = face_location

    return frame[y1:y2,x1:x2,:]





def refined_box(left, top, width, height):
    right = left + width
    bottom = top + height

    original_vert_height = bottom - top
    top = int(top + original_vert_height * 0.15)
    bottom = int(bottom - original_vert_height * 0.05)

    margin = ((bottom - top) - (right - left)) // 2
    left = left - margin if (bottom - top - right + left) % 2 == 0 else left - margin - 1

    right = right + margin

    return left, top, right, bottom



def apply_offsets_to_face_bounding_box(face_bounding_box, offsets, image_shape):
    '''
    apply offsets/margin on width and height of the bounding box

        paramters:
            face_bounding_box: bounding boxes of detected faces in the provided image
                                bounding boxes format (x1, y1, x2, y2)/(left, top, right, bottom)
                                where (x1, y1) is the upper left corner of the box,
                                        (x2, y2) is the lower right corner of the box.
            offset: tuple/list contains 2 values which is => [width margin, height margin]
            image_shape : shape of image in numpy format [height, width, channels] 
        
        return:
            new_bounding_box : the new bounding box after refining and margin applied. 

    '''
    x1, y1, x2, y2 = face_bounding_box
    x_off, y_off = offsets


    if x_off <= 1.0 or y_off<=1.0:
        x_off = int(x_off *image_shape[0]) # precentage of the height to be added to width
        y_off = int(y_off *image_shape[1])



    x1 = x1 - x_off if x1 - x_off >= 0 else 0
    y1 = y1 - y_off if y1 - y_off >= 0 else 0

    x2 = x2 + x_off if x2 + x_off < image_shape[1] else image_shape[1]-1
    y2 = y2 + y_off if y2 + y_off < image_shape[0] else image_shape[0]-1


    new_bounding_box = [int(x1), int(y1), int(x2), int(y2)]
    
    return new_bounding_box


def preprocess_input(x, v2=True):
    x = x.astype('float32')
    x = x / 255.0
    if v2:
        x = x - 0.5
        x = x * 2.0
    return x




def get_labels(dataset_name):
    if dataset_name == 'fer2013':
        return {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy',
                4: 'sad', 5: 'surprise', 6: 'neutral'}
    elif dataset_name == 'imdb':
        return {0: 'woman', 1: 'man'}
    elif dataset_name == 'KDEF':
        return {0: 'AN', 1: 'DI', 2: 'AF', 3: 'HA', 4: 'SA', 5: 'SU', 6: 'NE'}
    else:
        raise Exception('Invalid dataset name')


def rotate_box(box, angle, center=None):

    '''
    rotate bounding box by given angle
        parameters :
            box : python list of 2d points contain box 4 corners points
            angle : float number , the rotation angle to rotate the box
            center point to rotate the box around it, if none the center of box will be used
        
        returns:
            [p1, p2, p3, p4]: python list contains the four 2d corner points of the new rotated box

    '''

    # rotated box point [bottom left, top left, top right, bottom right]
    x_start, y_start, x_end, y_end = box
    p1 = [x_start, y_end]   # bottom left   
    p2 = [x_start, y_start] # top left
    p3 = [x_end, y_start]   # top right
    p4 = [x_end, y_end]     # bottom right


    if center == None:
        center = [x_start+ ((x_end-x_start)//2), y_start+ ((y_end-y_start)//2)]

    box_points = [p1,p2,p3,p4]

    rotated_box_points = rotate_polygon_around_point(box_points, center, angle)

    p1, p2, p3, p4 = rotated_box_points

    return [p1, p2, p3, p4]




def rotate_polygon_around_point(polygon, centerPoint, angle):
    '''
    rotate plygon points around specific point.
        parameters: 
            polygon : python list of 2D points hold polygon points.
            centerPoint : 2d point to rotate polygon points around it.
            angle: the float rotation angle to rotate the polygon 
        returns:
            polygon: the polygon points after being rotated.
    '''
    for i in range(len(polygon)):
        polygon[i] = rotatePoint(centerPoint,polygon[i],angle)
    
    return polygon

   

def rotatePoint(centerPoint,point,angle):
    """
    Rotates a point around another centerPoint. Angle is in degrees.
    Rotation is counter-clockwise.

        parameters:
            centerPoint : 2d point to rotate point around it.
            point: 2d point to be rotated.
            angle: the float rotation angle to rotate the polygon 
        
        returns:
            rotated point

    """
    angle = math.radians(angle)
    temp_point = point[0]-centerPoint[0] , point[1]-centerPoint[1]
    temp_point = ( temp_point[0]*math.cos(angle)-temp_point[1]*math.sin(angle) , temp_point[0]*math.sin(angle)+temp_point[1]*math.cos(angle))
    temp_point = int(temp_point[0]+centerPoint[0]) , int(temp_point[1]+centerPoint[1])
    return temp_point





class FPS:
    def __init__(self):
        # store the start time, end time, and total number of frames
        # that were examined between the start and end intervals
        self._start = None
        self._end = None
        self._num_frames = 0

    def start(self):
        self._start = time.time()

    def stop(self):
        self._end = time.time()

    def update(self):
        # increment the total number of frames examined during the
        # start and end intervals
        self._num_frames += 1

    def frames(self):
        return self._num_frames

    def elapsed(self):
        # return the total number of seconds between the start and
        # end interval
        if self._end == None:
            self._end=time.time()
        return self._end - self._start

    def fps(self):
        # compute the (approximate) frames per second
        return self._num_frames / self.elapsed()
