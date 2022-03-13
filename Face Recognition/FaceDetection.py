import cv2
import numpy as np
face_cascade=cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
# cap=cv2.VideoCapture("http://192.168.43.214:80/stream")
cap=cv2.VideoCapture(0)
cap.set(3,640)
cap.set(4,480)

while True:
    ret,img=cap.read()
    # Below line is used to flip an image around y- axix
    img=cv2.flip(img,1)
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,     
        scaleFactor=1.2,
        minNeighbors=5,     
        minSize=(20, 20)
    )
    
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
    
    cv2.imshow('video',img)
    k=cv2.waitKey(30)
    if k==27:
        break
cap.release()
cv2.destroyAllWindows()
