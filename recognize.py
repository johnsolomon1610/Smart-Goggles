import numpy as np
import pickle
import os
from cv2 import cv2
import time
import datetime
import imutils
import pyttsx3
from gtts import gTTS

curr_path = os.getcwd() #return current path

#face detection module initialization
print("Loading face_detection model........")
proto_path = os.path.join(curr_path, 'model', 'deploy.prototxt')
model_path = os.path.join(curr_path, 'model', 'res10_300x300_ssd_iter_140000.caffemodel')
face_detector = cv2.dnn.readNetFromCaffe(prototxt=proto_path, caffeModel=model_path)

print("Loading face_recognition model")
recognition_model = os.path.join(curr_path, 'model', 'openface_nn4.small2.v1.t7')
face_recognizer = cv2.dnn.readNetFromTorch(model=recognition_model)

recognizer = pickle.loads(open('recognizer.pickle', "rb").read())
le = pickle.loads(open('le.pickle', "rb").read())

print("Starting webcamera")
vs = cv2.VideoCapture(0)
time.sleep(1)


start_time=time.time()
ip_sec=8

while True:

    ret, frame = vs.read()
    frame = imutils.resize(frame, width=600)

    (h, w) = frame.shape[:2]

    image_blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0), False, False)

    face_detector.setInput(image_blob)
    face_detections = face_detector.forward()

    for i in range(0, face_detections.shape[2]):
        confidence = face_detections[0, 0, i, 2]

        if confidence >= 0.6:
            box = face_detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            face = frame[startY:endY, startX:endX]

            (fH, fW) = face.shape[:2]

            face_blob = cv2.dnn.blobFromImage(face, 1.0/255, (96, 96), (0, 0, 0), True, False)

            face_recognizer.setInput(face_blob)
            vec = face_recognizer.forward()

            preds = recognizer.predict_proba(vec)[0]
            j = np.argmax(preds) #returns max value along axes
            proba = preds[j]
            name = le.classes_[j]

            text = "{}: {:.2f}".format(name, proba * 100)
            y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)
            cv2.putText(frame, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
            
    

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    #time based camera operation

    curr_time=time.time()
    elap_time=curr_time-start_time

    if elap_time>ip_sec:
        print('recog successful in : '+str(int(elap_time)))
        break
    if key == ord('q'):
        break

texts=text.split(':')
texts=texts[0] 
speakcmd=pyttsx3.init()
speakcmd.say(texts)
speakcmd.runAndWait()
print(texts)
#send this texts to add person when unknown
cv2.destroyAllWindows()