import cv2
from matplotlib import pyplot as plt

face_cascade = cv2.CascadeClassifier('modelo/haarcascade_frontalface_default.xml')

print(cv2.__version__)
vidcap = cv2.VideoCapture('imagem/video.mp4')
success,img = vidcap.read()
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
count = 0
success = True
cont = 0
faces = face_cascade.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=5,
    minSize=(30, 30),
)
for (x,y,w,h) in faces:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  
    img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]
    imagem = 'imagem/serie_face_' + str(cont) + '.jpg'
    cv2.imwrite(imagem, roi_color)
    cont=cont + 1
  
cv2.imwrite("Faces/frame%d.jpg" % count, img)     # save frame as JPEG file
success,image = vidcap.read()
count += 1