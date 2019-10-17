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
    plt.hist(img.ravel(),255,[0,255])
    histg = cv2.calcHist([img],[0],None,[255],[0,255])
    plt.plot(histg)
  #plt.savefig("Frame%d.jpg" % count, img)
    plt.savefig("FaceHist/face_hito"+str(count)+".png")
    plt.clf()
    cont=cont + 1
