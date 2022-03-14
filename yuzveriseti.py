import cv2

kamera = cv2.VideoCapture(0)
kamera.set(3, 640)
kamera.set(4, 480)
face_detector = cv2.CascadeClassifier('D:\PyCharm\Yuz_Tanima\Cascade\haarcascade-frontalface-default.xml')

maxfotosayisi = 50 #Her bir yüz için kullanılan fotoğraf sayısı
say = 0
print("\nKayıtlar başlıyor. Lütfen bekleyiniz.")
face_id = input('\nID numarasını giriniz:') #Her farklı kişi için farklı bir yüz tam sayısı atayın

while(True):
    ret, img = kamera.read()
    img = cv2.flip(img, +1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    yuzler = face_detector.detectMultiScale(gray, 1.3, 5)

    for(x, y, w, h) in yuzler:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)
        say += 1
        cv2.imwrite("VeriSeti/" + str(face_id) + '.' + str(say) + ".jpg", gray[y:y+h, x:x+w]) #okunan yüzü veriseti klasörüne kaydeder
        cv2.imshow('resim', img)
        print("Kayıt no: ", say)

    k = cv2.waitKey(100) & 0xff

    if k == 27:
         break
    elif say >= maxfotosayisi:
        break
print("\nProgram sonlanıyor ve bellek temizleniyor.")
kamera.release()
cv2.destroyAllWindows()


