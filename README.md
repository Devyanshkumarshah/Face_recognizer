# Face_recognizer

Project Overview: Face Recognition using LBPH and Haar Cascades

This project demonstrates a basic face recognition system using OpenCV's LBPH (Local Binary Patterns Histograms) face recognizer and Haar cascades for face detection. The process involves two main phases: training and recognition.

Phase 1: Training the Model (face_recog.py)

Data Collection: Gather images of the individuals you want to recognize. Organize these images into separate folders, one folder per person. This creates a labeled dataset.

Face Detection (using Haar Cascades):

Load the pre-trained Haar cascade classifier (haar_face.xml). This classifier is trained to detect faces in images.
For each image in your dataset:
Convert the image to grayscale (often improves performance and reduces computation).
Use the Haar cascade classifier to detect faces in the grayscale image. This will give you bounding boxes (x, y, width, height) around each detected face.
Extract the Region of Interest (ROI) – the actual face area – from the image using the bounding box coordinates.
Feature Extraction (using LBPH):

For each detected face (ROI), apply the LBPH algorithm. LBPH calculates a histogram of local binary patterns, which represents the texture and appearance of the face. This histogram is the "feature vector" that represents the face.
Model Training:

Create an LBPH face recognizer object.
Train the LBPH recognizer by providing it with the extracted LBPH feature vectors (histograms) and their corresponding labels (the names of the people). The recognizer learns to associate these feature vectors with specific identities.
Model Saving:

Save the trained LBPH face recognizer model to a .yml file (e.g., face_trained.yml). This file contains the learned parameters of the model, which are needed for recognition. You also saved the features and labels separately, which is good practice.
Phase 2: Face Recognition (using the Trained Model)

Model Loading:

Load the trained LBPH face recognizer from the .yml file (face_trained.yml).
Load the Haar cascade classifier (same as in training).
Load the people list (must be in the same order as during training).
Image Loading and Preprocessing:

Load the image you want to recognize faces in.
Convert the image to grayscale.
Face Detection:

Use the Haar cascade classifier to detect faces in the image.
Face Recognition:

For each detected face:
Extract the ROI (face area).
Use the loaded LBPH face recognizer to predict the identity of the face. The recognizer will return a label (index) and a confidence score.
Use the label to look up the name of the person in the people list.
Displaying Results:

Draw a rectangle around each detected face.
Display the predicted name next to each face.
Print the predicted name and confidence score to the console
