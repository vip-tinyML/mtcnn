import cv2
import numpy as np
from mtcnn import MTCNN

# Load the image
image_path = 'images/jakAndMoi.jpg'
image = cv2.imread(image_path)

detector = MTCNN()

face_data = detector.detect_faces(image)


for i in range(len(face_data)):
    # Extract face coordinates and keypoints
    face_box = face_data[i]['box']

    # Draw bounding box and keypoints on the image
    x, y, w, h = face_box
    cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)  # Draw face bounding box

# Display the result
cv2.imshow('image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
