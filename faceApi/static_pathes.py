
haarcascade_frontalface_Path = "./faceApi/models/OpenCV/haarcascade_frontalface_default.xml" # for cascade method

# 8 bit Quantized version using Tensorflow ( 2.7 MB )
TFmodelFile = "./faceApi/models/OpenCV/tensorflow/TF_faceDetectorModel_uint8.pb"
TFconfigFile = "./faceApi/models/OpenCV/tensorflow/TF_faceDetectorConfig.pbtxt"


# FP16 version of the original caffe implementation ( 5.4 MB )
CAFFEmodelFile = "./faceApi/models/OpenCV/caffe/FaceDetect_Res10_300x300_ssd_iter_140000_fp16.caffemodel"
CAFFEconfigFile = "./faceApi/models/OpenCV/caffe/FaceDetect_Deploy.prototxt"


dlib_mmod_model_Path = "./faceApi/models/Dlib/mmod_human_face_detector.dat" # for Dlib MMOD 

dlib_5_face_landmarks_path = './faceApi/models/Dlib/shape_predictor_5_face_landmarks.dat'

ultra_light_640_onnx_model_path = './faceApi/models/UltraLight/ultra_light_640.onnx'
ultra_light_320_onnx_model_path = './faceApi/models/UltraLight/ultra_light_320.onnx'


yoloV3FacePretrainedWeightsPath = './faceApi/models/Yolo/yolo_weights/yolov3-wider_16000.weights'
yoloV3FaceConfigFilePath = './faceApi/models/Yolo/yolo_models_config/yolov3-face.cfg'
yoloV3FaceClassesFilePath = './faceApi/models/Yolo/yolo_labels/face_classes.txt'


known_faces_encodes_npy_file_path = './faceApi/FACEDB/known_faces_encodes.npy'
known_faces_ids_npy_file_path= './faceApi/FACEDB/known_faces_ids.npy'

emotion_model_path = './faceApi/models/emotion_models/emotions_model.h5'
gender_model_path = './faceApi/models/gender_models/gender_model.h5'


faceTageModelPath = './faceApi/models/sklearn/face_model.pkl'
