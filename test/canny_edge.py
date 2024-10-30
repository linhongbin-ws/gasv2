import cv2

import numpy as np

# Load the image

image = cv2.imread('test_seg.png')

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Gaussian Blur to reduce noise

blurred = cv2.GaussianBlur(gray, (5, 5), 1.5)

# Perform Canny edge detection

edges = cv2.Canny(blurred, threshold1=30, threshold2=100)

# Find contours from the edges

contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Draw the closed contours on the original image

result = image.copy()

cv2.drawContours(result, contours, -1, (0, 255, 0), 3)

# Show the original image, edges, and the result

cv2.imshow('Original Image', image)

cv2.imshow('Canny Edges', edges)

cv2.imshow('Detected Closed Boundaries', result)

cv2.waitKey(0)

cv2.destroyAllWindows()