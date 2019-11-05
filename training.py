import cv2
import os
import numpy as np


#classifiers
eigenface = cv2.face.EigenFaceRecognizer_create()
fisherface = cv2.face.FisherFaceRecognizer_create()
lbph = cv2.face.LBPHFaceRecognizer_create()





def getImageId():
    paths = [os.path.join("photos", f) for f in os.listdir("photos")]
    faces = []
    ids = []

    for pathimages in paths:

        faceimage = cv2.cvtColor(cv2.imread(pathimages), cv2.COLOR_BGR2GRAY)

        id = int(os.path.split(pathimages) [-1].split('.')[1])
        ids.append(id)
        faces.append(faceimage)

    return np.array(ids), faces


ids, faces = getImageId()



print("Training... ")

#for training needs to have two different classes (two faces of different people)For training
eigenface.train(faces, ids)
eigenface.write('eigenclassifier.yml') #save the classifier


fisherface.train(faces, ids)
fisherface.write('fisherclassifier')

lbph.train(faces,ids)
lbph.write('lbphclassifier')

print("training completed")



