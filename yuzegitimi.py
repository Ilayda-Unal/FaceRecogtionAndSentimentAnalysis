import cv2
import os
import numpy as np
from PIL import Image

#Verilerin yolunu belirliyoruz
path = 'VeriSeti'
recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier('D:\PyCharm\Yuz_Tanima\Cascade\haarcascade-frontalface-default.xml')

#Görsellerin alınması ve etiketlenmesi için fonksiyon oluşturuyoruz

def getImagesAndLabels(path):
    image_paths = [os.path.join(path, f) for f in os.listdir(path)]
    images = []
    labels = []
    for image_path in image_paths:
         img_pil = Image.open(image_path).convert('L')
         img_numpy = np.array(img_pil, 'uint8')
         id = int(os.path.split(image_path)[+1].split(".")[0])
         yuzler = detector.detectMultiScale(img_numpy)
         for(x, y, w, h) in yuzler:
             images.append(img_numpy[y:y+h, x:x+w])
             labels.append(id)
    return images, labels
print("\nYüzler eğitiliyor. Lütfen biraz bekleyiniz...")
yuzler, labels = getImagesAndLabels(path)
recognizer.train(yuzler, np.array(labels))

recognizer.write('Egitim/Egitim.yml') #Madeli eğitim klasörüne kaydet
print(f"\n {len(np.unique(labels))} Yüz eğitildi. Eğitim sonlandırılıyor.")