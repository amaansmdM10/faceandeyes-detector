import cv2
#this file contains harscade frontalface xml file
path1 = 'E:/document/3-1/cppsecrets project/haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(path1)
#this file conatins harscade eye xml file
path2 = 'E:/document/3-1/cppsecrets project/haarcascade_eye.xml'
eyes_cascade = cv2.CascadeClassifier(path2)
#this path conatins video file
path_video = 'E:/document/3-1/cppsecrets project/loki.mp4'
cap = cv2.VideoCapture(path_video)
while True:
    ret,img = cap.read()
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray,1.3,2)
    #draw bounding boxes 
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h,x:x+w]
        roi_color = img[y:y+h,x:x+w]
        eyes = eyes_cascade.detectMultiScale(roi_gray,1.3,4)
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

    cv2.imshow("output",img)
    cv2.waitKey(1000//25)
cap.release()