import cv2
import numpy as np

facedetect=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
rec=cv2.face.LBPHFaceRecognizer_create()
rec.read('trainingdata.yml')

id=0
fontface=cv2.FONT_HERSHEY_SIMPLEX
fontscale=1
fontcolor=(255,0,255)

id_map=['Ojaswini','Rishika','Amitabh Bachhan','Sarita']
cam=cv2.VideoCapture(0)

while(True):
     ret,img=cam.read()
     gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)#in gray scale the errors or noises are less
     faces=facedetect.detectMultiScale(gray,1.3,5)
     for(x,y,w,h) in faces:
         cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
         id,conf=rec.predict(gray[y:y+h,x:x+w])#basically it is the accuracy

         cv2.putText(img,str(id_map[id-1])+'_'+str(conf),(x,y+h),fontface,fontscale,fontcolor)


     cv2.imshow('face',img)
     if(cv2.waitKey(1)==ord('q')):
         break
cam.release()
cv2.destroyAllWindows()
