# import the necessary packages
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream

#from pygame import mixer
import numpy as np
import imutils
import time
import cv2
import os
import math

# system libraries
import os
import sys
from threading import Timer
import shutil
import time

body_result = []
detections = np.array(2)

def body_predictor(frame, maskNet, threshold=0.5):
    # grab the dimensions of the frame and then construct a blob
    # from it
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))

    # initialize our list of body, their corresponding locations,
    # and the list of predictions from our body mask network
    bodies = []
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

            # extract the body ROI, convert it from BGR to RGB channel
            # ordering, resize it to 224x224, and preprocess it
            body = frame[startY:endY, startX:endX]
            body = cv2.cvtColor(body, cv2.COLOR_BGR2RGB)
            body = cv2.resize(body, (224, 224))
            body = img_to_array(body)
            body = preprocess_input(body)
            body = np.expand_dims(body, axis=0)

            # add the body and bounding boxes to their respective
            # lists
            locs.append((startX, startY, endX, endY))
            # print(maskNet.predict(body)[0].tolist())
            preds.append(maskNet.predict(body)[0].tolist())

    return (locs, preds, body_result)
