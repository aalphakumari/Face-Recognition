import cv2
import numpy as np

facedetect=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

sampleNum=0
uid=input('Enter the userid')

cam=cv2.VideoCapture(0)
while(True):
       ret,img=cam.read()
       gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)#in gray scale the errors or noises are less
       faces=facedetect.detectMultiScale(gray,1.3,5)#rectangle accross the face

       for (x,y,w,h) in faces:
              sampleNum+=1
              cv2.imwrite('dataset/'+str(uid)+'_'+str(sampleNum)+'.jpg',gray[y:y+h,x:x+w])#it is used for saving and then retrieving the code
              cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,255),2)
              cv2.waitKey(100)
       cv2.imshow('face',img)
       cv2.waitKey(0)
       if(sampleNum>50):
              break
cam.release()
cam.destroyAllWindows()
