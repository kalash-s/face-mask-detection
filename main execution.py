import cv2
import os
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import numpy as np
import winsound

freq = 1000
dur = 300
faceCascade = cv2.CascadeClassifier(
    r'C:\Users\DELL\Documents\GitHub\Face-Mask-Detection-Project\haarcascade_frontalface_default.xml')

model = load_model(r'C:\Users\DELL\Documents\GitHub\Face-Mask-Detection-Project\mask_recog_ver2.h5')

video_capture = cv2.VideoCapture(0)

while True:
    ret, frame = video_capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(gray,
                                         scaleFactor=1.1,
                                         minNeighbors=5,
                                         minSize=(60, 60),
                                         flags=cv2.CASCADE_SCALE_IMAGE)
    faces_list = []
    preds = []
# faces are now preprocessed so that they can be fed into the model for prediction
    for (x, y, w, h) in faces:
        face_frame = frame[y:y + h, x:x + w]
        face_frame = cv2.cvtColor(face_frame, cv2.COLOR_BGR2RGB)
        face_frame = cv2.resize(face_frame, (224, 224))
        face_frame = img_to_array(face_frame)
        face_frame = np.expand_dims(face_frame, axis=0)
        face_frame = preprocess_input(face_frame)
        faces_list.append(face_frame)
        if len(faces_list) > 0:
            preds = model.predict(faces_list)
        for pred in preds:
            (mask, withoutMask) = pred
        label = "Mask" if mask > withoutMask else "No Mask"

        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
        if mask < withoutMask:
            winsound.Beep(freq, dur)
        # for accuracy in floating point as percentage
        label = "{}: {:.2f}%".format(label, max(mask, withoutMask)*100)
        # puts  label of mask or no mask
        cv2.putText(frame, label, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        # puts rectangle around our face
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        # Display the resulting frame
    cv2.imshow('Video', frame)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
video_capture.release()
cv2.destroyAllWindows()






