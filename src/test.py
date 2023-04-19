import cv2
import numpy as np

# Load the image
img = cv2.imread('res/test1.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 50, 150, apertureSize=3, L2gradient=True)

# Apply morphological operations to close gaps and holes
kernel = np.ones((5,5), np.uint8)
closing = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

# Find contours of the binary image
contours, hierarchy = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Create a new binary image to hold the filled contours
mask = np.zeros_like(gray)

# Draw each contour individually
for i, cnt in enumerate(contours):
    cv2.drawContours(mask, [cnt], -1, (255, 255, 255), cv2.FILLED)

# Display the resulting image
cv2.imshow('Filled Shapes', mask)
cv2.waitKey(0)
cv2.destroyAllWindows()