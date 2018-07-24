import numpy as np
import cv2
facedetect=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cam=cv2.VideoCapture(0)
while True:
  ret,img=cam.read()
  gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
  faces=facedetect.detectMultiScale(gray,1.3,5)#it makes a image pyramid and 1.3 means reduces 30% of the image

  for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),((x+w),(y+h)),(0,255,0),2)
  cv2.imshow('face',img)
  if(cv2.waitKey(0)==ord('q')):
       break
cam.release()
cv2.destroyAllWindows()
