import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import cv2
from typing import Optional
import os

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
# print(len(x_train), len(x_test))
# print(x_train.shape, y_train.shape) # (60000, 28, 28) (60000,)
# print(x_test.shape, y_test.shape)   # (10000, 28, 28) (10000,)
# print(x_train[1]) 
# plt.imshow(x_train[7], cmap='gray')
# plt.show()
x_train_flattened = x_train.reshape((len(x_train), 28 * 28)).astype('float32') / 255
x_test_flattened = x_test.reshape((len(x_test), 28 * 28)).astype('float32') / 255

model_loaded = False
model_path = 'mnist_mlp.keras'
if os.path.exists(model_path):
    try:
        model = keras.models.load_model(model_path)
        model_loaded = True
        print(f"Loaded saved model from {model_path}. Skipping training to start webcam faster.")
    except Exception as e:
        print(f"Failed to load saved model ({e}). Will train a new one.")

if not model_loaded:
    model = keras.Sequential([
        keras.layers.Dense(128, activation='relu', input_shape=(28 * 28,)),
        keras.layers.Dense(10, activation='softmax')
        ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy']
    )

    model.fit(x_train_flattened, y_train, epochs=5)

    model.evaluate(x_test_flattened, y_test)

    y_pred = model.predict(x_test_flattened)
    print(np.argmax(y_pred[4]))  # print the predicted label for the first test image   
    plt.imshow(x_test[4], cmap='gray')
    plt.show()

    #cm = tf.math.confusion_matrix(labels=y_test, predictions=np.argmax(y_pred, axis=1))
    #print(cm)

    # Save the trained model for reuse
    model.save(model_path)

# --- Webcam real-time digit recognition ---

def _preprocess_frame_to_mnist_vector(frame: np.ndarray) -> Optional[np.ndarray]:
    """
    Convert a BGR webcam frame to a 28x28 MNIST-like flattened vector (float32, [0,1]).
    Returns None if no digit-like contour is found.
    """
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Blur to reduce noise
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # Adaptive threshold to get a binary image; invert so foreground is white (like MNIST)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Find contours and pick the largest
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    c = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)
    if w * h < 100:  # too small, likely noise
        return None

    # Extract ROI with some padding
    pad = max(4, int(0.15 * max(w, h)))
    H, W = gray.shape[:2]
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(W, x + w + pad)
    y2 = min(H, y + h + pad)
    roi = gray[y1:y2, x1:x2]

    # Normalize aspect ratio onto 20x20 then place centered in 28x28 (similar to classic MNIST processing)
    h_roi, w_roi = roi.shape
    if h_roi == 0 or w_roi == 0:
        return None
    scale = 20.0 / max(h_roi, w_roi)
    new_w = max(1, int(w_roi * scale))
    new_h = max(1, int(h_roi * scale))
    resized = cv2.resize(roi, (new_w, new_h), interpolation=cv2.INTER_AREA)

    canvas = np.zeros((28, 28), dtype=np.uint8)
    x_offset = (28 - new_w) // 2
    y_offset = (28 - new_h) // 2
    canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized

    # Invert colors to match training (training used raw MNIST where 0=black background, white strokes ~1)
    # Our thresholding produced white foreground on black background, which is consistent with MNIST.

    # Normalize to [0,1] and flatten
    canvas = canvas.astype('float32') / 255.0
    vec = canvas.reshape(28 * 28)
    return vec


def _run_webcam_inference():
    # Try DirectShow backend first (faster on Windows); fall back to default
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open webcam.")
        return

    print("Webcam started. Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        vec = _preprocess_frame_to_mnist_vector(frame)
        prediction_text = "No digit"
        if vec is not None:
            logits = model.predict(vec[None, :], verbose=0)
            pred = int(np.argmax(logits, axis=1)[0])
            probability = float(np.max(logits))
            prediction_text = f"Pred: {pred} ({probability:.2f})"

            # Draw bounding box for the largest contour for feedback
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(cv2.GaussianBlur(gray, (5, 5), 0), 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                c = max(contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(c)
                if w * h >= 100:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.putText(frame, prediction_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow('Digit Recognizer (press q to quit)', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# Start webcam inference after training/evaluation/visualization
_run_webcam_inference()