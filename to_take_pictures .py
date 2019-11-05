import cv2
#Captura a imagem da pessoa da webcam e atribui um identificador, após isso


classifier = cv2.CascadeClassifier("haarcascade-frontalface-default.xml") #Classifier with Haar Cascade Viola–Jones method
eyeclassifer = cv2.CascadeClassifier("haarcascade-eye.xml")
camera = cv2.VideoCapture(0) #Detect Camera
sample = 1 #variable to control how many picture were made
numbersamples = 25 #will be made 25 photos of each people
id = input("Type your name: ")
width, height = 220, 220 #control the size of the photo will be taken
#Fisherfaces e o Eigen Faces para treinamento necessita que o tamanho da imagem seja iguais.
print("Capturing faces....")

while (True):
    connected, image = camera.read()
    #make gray image detection the algorithm performance is better
    grayimage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #convert image to grayscale
    facesdetected =  classifier.detectMultiScale(grayimage, scaleFactor=1.5, minSize=(100,100)) #scale factor indicates the image scale

    for (x, y, w, h) in facesdetected: #w = width h=height
        cv2.rectangle(image, (x, y),(x + h, y + w ), (0, 0, 255), 2 ) #bouding box
        if cv2.waitKey(1) & 0xFF == ord('q'): #toda vez que apertar a tecla q ele executa o que está abaixo do if (salvar a imagem que está passando na webcam)
            faceImage = cv2.resize(grayimage[y:y + h, x:x + w], (width, height))
            cv2.imwrite("photos/person." + str(id) + "." + str(sample) + ".jpg", faceImage)
            print("[picture " + str(sample) + "taken]")
            sample += 1


    cv2.imshow("Face", image)
    cv2.waitKey(1)

    if (sample >= numbersamples + 1):
        break


camera.realease() #free the memory
cv2.DestroyAllWindows()


