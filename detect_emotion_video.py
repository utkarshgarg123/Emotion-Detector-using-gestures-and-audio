# import the necessary packages
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
from speech.speechemotionrecognition import model_classes
from speech.examples.Speech_reciever import Speech_reciever
# from pygame import mixer
import numpy as np
import imutils
import time
import cv2
import os
import sounddevice as sd
from scipy.io.wavfile import write


from fer import FER
from deepface import DeepFace
# importing codes
# system libraries
import os
import sys
from threading import Timer
import shutil
import time


def detect_and_predict_mask(frame, faceNet, maskNet, threshold):
    # grab the dimensions of the frame and then construct a blob
    # from it
    global detections
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))

    # pass the blob through the network and obtain the face detections
    faceNet.setInput(blob)
    detections = faceNet.forward()

    # initialize our list of faces, their corresponding locations,
    # and the list of predictions from our face mask network
    faces = []
    locs = []
    preds = []
    # loop over the detections
    for i in range(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with
        confidence = detections[0, 0, i, 2]

        # filter out weak detections by ensuring the confidence is
        # greater than the minimum confidence
        if confidence > threshold:
            # compute the (x, y)-coordinates of the bounding box for
            # the object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # ensure the bounding boxes fall within the dimensions of
            # the frame
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            # extract the face ROI, convert it from BGR to RGB channel
            # ordering, resize it to 224x224, and preprocess it
            face = frame[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)
            face = np.expand_dims(face, axis=0)

            # add the face and bounding boxes to their respective
            # lists
            locs.append((startX, startY, endX, endY))
            # print(maskNet.predict(face)[0].tolist())
            preds.append(maskNet.predict(face)[0].tolist())
    return (locs, preds)



def weighted_average(speech_result, result):
    temp = {'emotion': {'angry': 0, 'disgust': 0, 'fear': 0, 'happy': 0, 'sad': 0, 'surprise': 0 , 'neutral': 0 }, 'dominant_emotion': 'neutral'}
    temp['emotion']['angry'] = 0.1*speech_result['emotion']['angry'] + 0.9* result['emotion']['angry'] 
    temp['emotion']['disgust'] = 0.1*speech_result['emotion']['disgust'] + 0.9* result['emotion']['disgust'] 
    temp['emotion']['fear'] = 0.1*speech_result['emotion']['fear'] + 0.9* result['emotion']['fear'] 
    temp['dominant_emotion'] = result['dominant_emotion']
    temp['emotion']['happy'] = 0.1*speech_result['emotion']['happy'] + 0.9* result['emotion']['happy'] 
    temp['emotion']['sad'] = 0.1*speech_result['emotion']['sad'] + 0.9* result['emotion']['sad'] 
    temp['emotion']['neutral'] = 0.1*speech_result['emotion']['neutral'] + 0.9* result['emotion']['neutral'] 
    temp['emotion']['surprise'] = 0.1*speech_result['emotion']['surprise'] + 0.9* result['emotion']['surprise']
    return temp

    
    

# SETTINGS
MASK_MODEL_PATH = os.getcwd()+"\\model\\emotion_model.h5"
FACE_MODEL_PATH = os.getcwd()+"\\face_detector"
SOUND_PATH = os.getcwd()+"\\sounds\\alarm.wav"
THRESHOLD = 0.5

# Load Sounds
# mixer.init()
# sound = mixer.Sound(SOUND_PATH)

# load our serialized face detector model from disk
print("[INFO] loading face detector model...")
prototxtPath = os.path.sep.join([FACE_MODEL_PATH, "deploy.prototxt"])
weightsPath = os.path.sep.join(
    [FACE_MODEL_PATH, "res10_300x300_ssd_iter_140000.caffemodel"])
# faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# load the face mask detector model from disk
print("[INFO] loading emotion detector model...")
maskNet = load_model(MASK_MODEL_PATH)

# initialize the video stream and allow the camera sensor to warm up
print("[INFO] !!! starting video stream camera ...")

faceCascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture()

if not cap.isOpened():
    cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("Cannot open Webcam")

while True:
    ret, frame = cap.read()

# main function
    detector = FER()
    detector.detect_emotions(frame)
    result = DeepFace.analyze(frame, actions=['emotion'])
#


    # result =  face_processor(frame, faceNet, maskNet,threshold)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # (locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet, 0.5)

        
    
    #Sound recoder and analyser
    
    fs = 44100  # Sample rate
    seconds = 0.5  # Duration of recording

    myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=2)
    sd.wait()  # Wait until recording is finished
    speech_result = Speech_reciever(myrecording)
    ##########

    faces = faceCascade.detectMultiScale(gray, 1.1, 4)

    for(x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.rectangle(frame, (x-150, y-10), (x+w+200, y+h+200), (0, 255, 0), 2)

    # cv2.rectangle(frame, (50, 50), (100, 100), (0, 255, 0), 2)

    font = cv2.FONT_HERSHEY_SIMPLEX

     # print(result)
    final_result = weighted_average(speech_result,result)
    print('[DETECTED EMOTION] : ' , final_result['emotion'])
    print('[DOMINANT EMOTION] : ' , final_result['dominant_emotion'] , ' ' , final_result['emotion'][final_result['dominant_emotion']])


    cv2.putText(frame,
                final_result['dominant_emotion'],
                (50, 50),
                font, 3,
                (0, 0, 255),
                2,
                cv2.LINE_4)
    txt = str(final_result['emotion'][final_result['dominant_emotion']])
    cv2.putText(frame,
                txt,
                (70, 100),
                font, 1,
                (0, 20, 150),
                2,
                cv2.LINE_4)
    

   



    cv2.imshow('Original Video', frame)
    if cv2.waitKey(2) & 0xFF == ord('q'):
        break



cap.release()
cv2.destroyAllWindows()