import cv2
import numpy as np
from mtcnn import MTCNN

# Load the image
image_path = 'images/jakAndMoi.jpg'
image = cv2.imread(image_path)

detector = MTCNN()
face_data = detector.detect_faces(image)

# Define the detected face data (replace this with your model's output)
# face_data = [{'box': [829, 148, 292, 346], 'confidence': 0.9999959468841553, 'keypoints': {'left_eye': (899, 309), 'right_eye': (1022, 294), 'nose': (953, 389), 'mouth_left': (918, 417), 'mouth_right': (1044, 403)}}, {'box': [249, 177, 231, 294], 'confidence': 0.9999141693115234, 'keypoints': {'left_eye': (321, 283), 'right_eye': (427, 279), 'nose': (382, 322), 'mouth_left': (333, 396), 'mouth_right': (436, 387)}}]


for i in range(len(face_data)):
    # Extract face coordinates and keypoints
    face_box = face_data[i]['box']
    left_eye = face_data[i]['keypoints']['left_eye']
    right_eye = face_data[i]['keypoints']['right_eye']
    nose = face_data[i]['keypoints']['nose']
    mouth_left = face_data[i]['keypoints']['mouth_left']
    mouth_right = face_data[i]['keypoints']['mouth_right']

    # Draw bounding box and keypoints on the image
    x, y, w, h = face_box
    cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)  # Draw face bounding box

    # Draw keypoints (eyes, nose, mouth) as circles
    cv2.circle(image, left_eye, 5, (0, 255, 0), -1)      # Left eye
    cv2.circle(image, right_eye, 5, (0, 255, 0), -1)     # Right eye
    cv2.circle(image, nose, 5, (0, 255, 0), -1)           # Nose
    cv2.circle(image, mouth_left, 5, (0, 255, 0), -1)     # Left corner of the mouth
    cv2.circle(image, mouth_right, 5, (0, 255, 0), -1)    # Right corner of the mouth

# Display the result
cv2.imshow('Image with keypoints', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
