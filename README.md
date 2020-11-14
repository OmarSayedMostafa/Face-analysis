# Face-analysis
Face detection, recognition, gender classification, emotion classification

![Alt Text](faceApi/results/result.gif)

## Features:
  - multiple Face Detectors
  - Face Recognition
  - faciallandmarks detection
  - race Classification
  - gender Classification
  - emotion Detection ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']


## Notes:
the process of analysis is kinda slow because i use seperate models in this pipeline and it needs some optimizations like:
> - models quantization and using tensorRT For speeding the models
> - most of the faces models (age, gender, race, emotions) is using the same backbone and it can be groubed into one model that need to be retrained again with available datasets.
> - recognition process is applyed for each frame (used as face tracker) and it can be replaced with a suitable object tracker.
