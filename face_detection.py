import cv2
import numpy as np
from mtcnn import MTCNN

# Initialize MTCNN for face detection
detector = MTCNN()

# Initialize video capture (webcam)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Unable to open the webcam.")
    exit()

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()
    if not ret:
        print("Error: Unable to read the frame.")
        break

    # Convert the frame to RGB (required for MTCNN)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect faces
    results = detector.detect_faces(rgb_frame)

    # Loop through detected faces
    for result in results:
        x, y, w, h = result['box']  # Bounding box coordinates
        confidence = result['confidence']  # Model confidence

        if confidence > 0.9:  # Only if the confidence is high
            # Draw a rectangle around the face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Add a label with the confidence (in %)
            label = f"Conf: {confidence * 100:.1f}%"
            cv2.putText(frame, label, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the frame with rectangles
    cv2.imshow('Face Detection with MTCNN', frame)

    # Exit the loop if the user presses 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()