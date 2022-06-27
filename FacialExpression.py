from keras.models import load_model
from keras.preprocessing.image import img_to_array
import cv2
import numpy as np
import pandas as pd
import datetime
import os

def create_folder(name):
    fearName = str(name) + '/data/fear'
    happyName =  str(name) + '/data/happy'
    sadName = str(name) + '/data/sad'
    neutralName = str(name) + '/data/neutral'
    surpriseName = str(name) + '/data/surprise'
    disgustName = str(name) + '/data/disgust'
    angryName =  str(name) + '/data/angry'
    os.makedirs(fearName)
    os.makedirs(happyName)
    os.makedirs(sadName)
    os.makedirs(neutralName)
    os.makedirs(surpriseName)
    os.makedirs(disgustName)
    os.makedirs(angryName)

def main_app(name):

    face_classifier = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')
    classifier = load_model('./model_little_vgg.h5')

    class_labels = ['Angry','Disgust','Fearful','Happy','Neutral','Sad','Surprised']

    cap = cv2.VideoCapture(0)
    face_label = []
    preds_max = []
    preds_face = []
    time_label = []
    angry_frame = []
    happy_frame = []
    neutral_frame = []
    sad_frame = []
    surprise_frame = []
    disgust_frame = []
    fear_frame = []
    save_width = 48
    save_height = 48
    count = 0

    while True:
        #Grab a single frame of video
        ret, frame = cap.read()
        labels = []
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray,1.3,5)

        for (x,y,w,h) in faces:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(255,255,255),2)
            roi_gray = gray[y:y+h,x:x+w]
            roi_gray = cv2.resize(roi_gray,(48,48),interpolation=cv2.INTER_AREA)

            if np.sum([roi_gray])!=0:
                roi = roi_gray.astype('float')/255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi,axis=0)

            #make a prediction on the ROI, then  lookup the class

                preds = classifier.predict(roi)[0]
                print("\nprediction = ",preds)
                label = class_labels[preds.argmax()]
                print("\nprediction max = ",preds.argmax())
                print("\nlabel = ",label)
                count+=1
                print(count)
                if(label == 'Angry'):
                        angry_frame.append(frame)
                elif(label == 'Happy'):
                        happy_frame.append(frame)
                elif(label == 'Neutral'):
                        neutral_frame.append(frame)
                elif(label == 'Sad'):
                        sad_frame.append(frame)
                elif(label == 'Surprised'):
                        surprise_frame.append(frame)
                elif(label == 'Disgust'):
                    disgust_frame.append(frame)
                elif(label == 'Fearful'):
                    fear_frame.append(frame)
                label_position = (x,y)
                cv2.putText(frame,label,label_position,cv2.FONT_HERSHEY_SIMPLEX,2,(255,255,255))
                datetime_object = datetime.datetime.now()
                print("\nTime = ",datetime_object)
                face_label.append(label)
                preds_max.append(preds.argmax())
                preds_face.append(preds)
                time_label.append(datetime_object)
            else:
                cv2.putText(frame,'No Face Found',(20,60),cv2.FONT_HERSHEY_SIMPLEX,2,(0,0,255))
            print("n\n")
        cv2.imshow('Emotion Detector',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    #below is for saving the data of real-time face emotional detection to csv file
    data = list(zip(face_label,preds_max,preds_face,time_label))

    df = pd.DataFrame(data, columns=["labels","classes","predictions","time"])

    file_path = str(name) + '/data'

    df.to_csv(file_path + '/data.csv',index=False,header=True)

    #below is for saving the image for every class_labels
    for i,frame in enumerate(angry_frame):
        roi = frame[75 + 2:425 - 2, 300 + 2:650 - 2]
        roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        roi = cv2.resize(roi, (save_width, save_height))
        cv2.imwrite(str(name) + '/data/angry/{}.png'.format(i), cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY))

    for i,frame in enumerate(happy_frame):
        roi = frame[75 + 2:425 - 2, 300 + 2:650 - 2]
        roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        roi = cv2.resize(roi, (save_width, save_height))
        cv2.imwrite(str(name) + '/data/happy/{}.png'.format(i), cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY))

    for i,frame in enumerate(neutral_frame):
        roi = frame[75 + 2:425 - 2, 300 + 2:650 - 2]
        roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        roi = cv2.resize(roi, (save_width, save_height))
        cv2.imwrite(str(name) + '/data/neutral/{}.png'.format(i), cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY))

    for i,frame in enumerate(sad_frame):
        roi = frame[75 + 2:425 - 2, 300 + 2:650 - 2]
        roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        roi = cv2.resize(roi, (save_width, save_height))
        cv2.imwrite(str(name) + '/data/sad/{}.png'.format(i), cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY))

    for i,frame in enumerate(surprise_frame):
        roi = frame[75 + 2:425 - 2, 300 + 2:650 - 2]
        roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        roi = cv2.resize(roi, (save_width, save_height))
        cv2.imwrite(str(name) + '/data/surprise/{}.png'.format(i), cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY))

    for i,frame in enumerate(disgust_frame):
        roi = frame[75 + 2:425 - 2, 300 + 2:650 - 2]
        roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        roi = cv2.resize(roi, (save_width, save_height))
        cv2.imwrite(str(name) + '/data/disgust/{}.png'.format(i), cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY))

    for i,frame in enumerate(fear_frame):
        roi = frame[75 + 2:425 - 2, 300 + 2:650 - 2]
        roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        roi = cv2.resize(roi, (save_width, save_height))
        cv2.imwrite(str(name) + '/data/fear/{}.png'.format(i), cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY))