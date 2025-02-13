import os
import cv2 as cv
import numpy as np 

people = ['Depp', 'modi', 'trump'] ## List of names of people to recognize

DIR = r'C:\Users\Devyansh kumar\Documents\opencv' ## Directory containing the images

haar_cascade = cv.CascadeClassifier('haar_face.xml') # Load the Haar cascade for face detection
features = []  # List to store the extracted face features (images)
labels = []    # List to store the extracted face features (images)


def creat_train():
    for person in people:    # Iterate through each person
        path = os.path.join(DIR, person)     # Construct the path to the person's image directory
        label = people.index(person)       # Get the label (index) for the current person
 
        for img in os.listdir(path):       # Iterate through each image in the person's directory
            img_path = os.path.join(path, img)   # Construct the full path to the image
 
            img_array = cv.imread(img_path)     # Read the image
            gray = cv.cvtColor(img_array, cv.COLOR_BGR2GRAY)    #Convert the image to grayscale

            faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor = 1.1, minNeighbors = 4)  # Detect faces

            for(x,y,w,h) in faces_rect: # Iterate through each detected face
                faces_roi = gray[y:y+h, x:x+w]   # Extract the face region of interest (ROI)
                features.append(faces_roi)    # Add the face ROI to the features list
                labels.append(label)    # Add the corresponding label to the labels list

creat_train()   # Call the function to create the training data
print('Training done--------------')

features = np.array(features, dtype='object')   # Convert the features list to a NumPy array
labels = np.array(labels)   # Convert the labels list to a NumPy array

face_recognizer = cv.face.LBPHFaceRecognizer_create()  # Create an LBPH face recognizer object

#Train the Recognizer on the features list and labels list

face_recognizer.train(features, labels)  # Train the face recognizer

face_recognizer.save('face_trained.yml')  # Save the trained face recognizer model
np.save('features.npy', features)  # Save the features array
np.save('labels.npy', labels)   # Save the labels array