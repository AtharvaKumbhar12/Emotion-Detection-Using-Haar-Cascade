import streamlit as st
from keras.models import load_model
from time import sleep
import tensorflow
from tensorflow.keras.utils import img_to_array
from keras.preprocessing import image
import cv2
import numpy as np


st.title('Emotion Detection')
face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
classifier =load_model('model.h5')
emotion_labels = ['Angry','Disgust','Fear','Happy','Neutral', 'Sad', 'Surprise']

cap = cv2.VideoCapture('test2.mp4')
frameST = st.empty()


n = 0

while True:
    if n%30 == 0:
        _, frame = cap.read()
        if not _:
            # print("Done processing !!!")
            st.text("Done Processing")
            #cv2.waitKey(3000)
            # Release device
            cap.release()
            break
        labels = []
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray)

        for (x,y,w,h) in faces:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)
            roi_gray = gray[y:y+h,x:x+w]
            roi_gray = cv2.resize(roi_gray,(48,48),interpolation=cv2.INTER_AREA)



            if np.sum([roi_gray])!=0:
                roi = roi_gray.astype('float')/255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi,axis=0)

                prediction = classifier.predict(roi)[0]
                label=emotion_labels[prediction.argmax()]
                label_position = (x,y)
                cv2.putText(frame,label,label_position,cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
                #frameST.image(frame)
            else:
                cv2.putText(frame,'No Faces',(30,80),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
                #frameST.image(frame)
        #cv2.imshow('Emotion Detector',frame)
        frameST.image(frame, channels="BGR")
        
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     cap.release()
        #     #break
    else:
        continue
cap.release()
cv2.destroyAllWindows()