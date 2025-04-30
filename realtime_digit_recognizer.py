import cv2
import numpy as np
from tensorflow.keras.models import load_model

# loadding
model = load_model("my_model.h5")

# Starts webcam
cap = cv2.VideoCapture(0)


x0, y0, x1, y1 = 100, 100, 300, 300

def preprocess(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    roi = gray[y0:y1, x0:x1]

   
    _, thresh = cv2.threshold(roi, 75, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Find the largest contour
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        cnt = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(cnt)
        digit = thresh[y:y+h, x:x+w]
    else:
        digit = thresh

    # Resize while keeping aspect ratio
    h, w = digit.shape
    if h > w:
        new_h, new_w = 20, int(w * (20 / h))
    else:
        new_w, new_h = 20, int(h * (20 / w))
    resized = cv2.resize(digit, (new_w, new_h))

    # Pad to 28x28
    padded = np.pad(resized,
        (((28-new_h)//2, (28-new_h)-(28-new_h)//2),
         ((28-new_w)//2, (28-new_w)-(28-new_w)//2)),
        mode='constant', constant_values=0)

    # Normalize and reshape
    normalized = padded.astype("float32") / 255.0
    reshaped = normalized.reshape(1, 28, 28, 1)
    return reshaped

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Draw ROI rectangle
    cv2.rectangle(frame, (x0, y0), (x1, y1), (255, 0, 0), 2)

    # Preprocess the digit
    digit_input = preprocess(frame)

    # Predict digit
    prediction = model.predict(digit_input)
    digit = np.argmax(prediction)

    # Display prediction
    cv2.putText(frame, f"Predicted: {digit}", (x0, y0 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show the webcam frame
    cv2.imshow("Digit Recognizer", frame)

    # Quit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
