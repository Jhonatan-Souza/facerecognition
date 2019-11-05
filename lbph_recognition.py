import cv2


faceDetector =  cv2.CascadeClassifier("haarcascade-frontalface-default.xml")
recognition = cv2.face.LBPHFaceRecognizer_create()
recognition.read("lbphclassifier.yml")
width, height = 220,220
font = cv2.FONT_HERSHEY_COMPLEX_SMALL

camera = cv2.VideoCapture(0)

while (True):
    connected, image = camera.read()
    grayimage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    detectedfaces = faceDetector.detectMultiScale(grayimage, scaleFactor=1.5, minSize=(30,30))

    for (x, y, w, h) in detectedfaces:
        faceimage = cv2.resize(grayimage[y:y + h, x:x + w], (width, height))
        cv2.rectangle(image, (x,y), (x + w, y + h), (0, 0, 255), 2)
        id, confidence = recognition.predict(faceimage)
        name = ""
        if id == 1:
            nome = 'Jhonatan'
        elif id == 2:
            nome = 'Anya'
        cv2.putText(image, nome, (x, y + (h + 30)), font, 2, (0, 0, 255))
        cv2.putText(image, str(confidence), (x, y + (h + 50)), font, 1, (0, 0, 255))



    cv2.imshow("Face", image)
    if cv2.waitKey(1) == ord ('q'):
        break



camera.release()
cv2.DestroyAllWindows()


