import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from keras.models import load_model
from time import sleep
from keras.preprocessing import image
from keras.preprocessing.image import img_to_array

model = load_model('D:\PyCharm\Yuz_Tanima\Cascade\Emotion_Detection.h5')

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('Egitim/Egitim.yml')
cascadePath = ('D:\PyCharm\Yuz_Tanima\Cascade\haarcascade-frontalface-default.xml')
faceCascade = cv2.CascadeClassifier(cascadePath)
font = cv2.FONT_HERSHEY_SIMPLEX

id = 0
adlar = ['None', 'Ilayda', 'Melisa', 'Onur', 'Yagmur']

model_labels = ['Kizgin', 'Mutlu', 'Notr', 'Uzgun', 'Saskin']

kamera = cv2.VideoCapture(0)  # Canlı görüntü yakalamayı başlatıyor.
kamera.set(3, 1000)
kamera.set(4, 1000)

minW = 0.1 * kamera.get(3)
minH = 0.1 * kamera.get(4)

while (True):
    ret, img = kamera.read()
    if not ret:
        continue

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    yuzler = faceCascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(int(minW), int(minH)), )

    for (x, y, w, h) in yuzler:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)

        id, uyum = recognizer.predict(gray[y:y + h, x:x + w])

        if (uyum < 100):
            id = adlar[id]
            uyum = f"Uyum= %{round(uyum, 0)}"
        else:
            id = "Bilinmiyor"
            uyum = f"Uyum= %{round(uyum, 0)}"

        color = (255, 255, 255)
        cv2.putText(img, str(id), (x + 5, y - 5), font, 1, color, 2)
        cv2.putText(img, str(uyum), (x + 5, y + h + 25), font, 1, color, 2)

    for (x, y, w, h) in yuzler:

        roi_gray = gray[y:y + h, x:x + w]
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

        if np.sum([roi_gray]) != 0:
            roi = roi_gray.astype('float') / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)

            preds = model.predict(roi)[0]
            print("\nprediction = ", preds)
            label = model_labels[preds.argmax()]
            print("\nprediction max = ", preds.argmax())
            print("\nlabel = ", label)
            label_position = (x, y)

            cv2.putText(img, str(label), (x + 200, y + 0), font, 1, (200, 0, 0), 3)

        cv2.imshow('Kamera', img)

    k = cv2.waitKey(10) & 0xff  # Programdan çıkış esc ya da q tuşu
    if k == 27 or k == ord('q'):
            break

print("\n Programdan çıkılıyor ve bellek temizleniyor")
kamera.release()
cv2.destroyAllWindows()