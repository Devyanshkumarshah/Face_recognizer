#Object Recognition: Shape Analysis: By identifying edges, you can extract features like corners, curves, and lines, which are essential for recognizing objects.
#Pattern Matching: Edge detection can be used to find specific patterns or shapes within an image, aiding in tasks like barcode scanning or character recognition.
import cv2 as cv 
import numpy as np
img = cv.imread('photos/cat.jpeg')
cv.imshow('Cats', img)

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Gray', gray)

#Laplacian: The Laplacian function in OpenCV is primarily used for edge detection in images. It's a powerful tool that helps identify 
#areas of significant intensity change, which often correspond to the boundaries of objects within an image.

lap = cv.Laplacian(gray, cv.CV_64F)
lap = np.uint8(np.absolute(lap))
cv.imshow('Laplacian', lap)

#sable
sobelx = cv.Sobel(gray, cv.CV_64F, 1, 0)
sobely = cv.Sobel(gray, cv.CV_64F, 0, 1)
combined_sobel = cv.bitwise_or(sobelx, sobely)

cv.imshow('sobel X', sobelx)
cv.imshow('sobel Y', sobely)

#combined Sobel
cv.imshow('combined Sobel', combined_sobel)

#coparing both sobel and laplacian image by canny edge detector

canny = cv.Canny(gray, 150, 175)
cv.imshow('canny', canny)


cv.waitKey(0)

 
