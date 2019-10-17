import cv2
import numpy as np
from matplotlib import pyplot as plt

vidcap = cv2.VideoCapture('imagem/video.mp4')
success,img = vidcap.read()
count = 0
success = True
while success:
  success,img = vidcap.read()  
       # save frame as JPEG file
  #img = img[::2,::2] # Diminui a imagem
  plt.hist(img.ravel(),256,[0,256])
  histg = cv2.calcHist([img],[0],None,[256],[0,256])
  plt.plot(histg)
  #plt.savefig("Frame%d.jpg" % count, img)
  plt.savefig("Hist/hito"+str(count)+".png")
  plt.clf()
  #cv2.imwrite("frame%d.jpg" % count, img)
  #plt.show()
  #break
  #Proxima fase, detectar a face, extrair ela e borra-la.
  #Necessário tirar o break para salvar todas.
  #cv2.imshow("Imagem original e suavisadas pela mediana", imagem_filtrada)
  #cv2.waitKey(0)
  
  count += 1