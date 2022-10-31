import cv2
import numpy as np

face_cas = cv2.CascadeClassifier('haarcascade_frontalface.xml')

def video(source=0):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        return False
    else:
        return cap

cap = video()
while cap.isOpened():
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cas.detectMultiScale(gray,
                                      scaleFactor=1.1,
                                      minNeighbors=5)
    for x, y, w, h in faces:
        frame = cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 6)
    cv2.imshow('detected', frame)
    if cv2.waitKey(1) & 0xff == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
