import numpy as np
import cv2 as cv 

haar_cascade = cv.CascadeClassifier('haar_face.xml')  # Load the Haar cascade for face detection

people = ['Depp', 'modi', 'trump']  # List of names (must match training order)
#features = np.load('features.npy', allow_pickle=True)
#labels = np.load('labels.npy')

face_recognizer = cv.face.LBPHFaceRecognizer_create()  # Create LBPH recognizer object
face_recognizer.read('face_trained.yml')  # Load the trained model

img = cv.imread(r'C:\Users\Devyansh kumar\Documents\opencv\Depp\download (1).jpeg')  # Load the image which you want to identify

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)# Convert to grayscale
cv.imshow('Person', gray)   # Display the grayscale image 

#detect the face in the image
faces_rect = haar_cascade.detectMultiScale(gray, 1.1, 4)  # Detect faces

for(x,y,w,h) in faces_rect:# Loop through each detected face
    faces_roi = gray[y:y+h, x:x+h] # Extract the face region (ROI) 

    label, confidence = face_recognizer.predict(faces_roi)  # Predict label and confidence
    print(f'Label = {people[label]} with a confidence of {confidence}')  # Print results

    cv.putText(img, str(people[label]), (20,20), cv.FONT_HERSHEY_COMPLEX, 1.0, (0,255,0), thickness=2)# Draw name
    cv.rectangle(img, (x,y), (x+w,y+h), (0,255,0), thickness=2) # Draw rectangle

    cv.imshow('Detected Face', img) # Display the image with results

    cv.waitKey(0)