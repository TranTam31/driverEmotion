# import cv2
# import numpy as np
# import time
# from mtcnn.mtcnn import MTCNN
# from keras.models import load_model
# from statistics import mode
# from utils.datasets import get_labels
# from utils.inference import detect_faces
# from utils.inference import draw_text
# from utils.inference import draw_bounding_box
# from utils.inference import apply_offsets
# from utils.inference import load_detection_model
# from utils.preprocessor import preprocess_input

# USE_WEBCAM = True # If false, loads video file source

# # parameters for loading data and images
# emotion_model_path = './models/emotion_model.hdf5'
# emotion_labels = get_labels('fer2013')

# # hyper-parameters for bounding boxes shape
# frame_window = 10
# emotion_offsets = (20, 40)

# last_prediction_time = time.time()
# prediction_interval = 1.0
# latest_emotions = []

# # loading models
# face_cascade = cv2.CascadeClassifier('./models/haarcascade_frontalface_default.xml')
# emotion_classifier = load_model(emotion_model_path)

# # getting input model shapes for inference
# emotion_target_size = emotion_classifier.input_shape[1:3]

# # starting lists for calculating modes
# emotion_window = []

# # starting video streaming

# cv2.namedWindow('window_frame')
# video_capture = cv2.VideoCapture(0)

# # Select video or webcam feed
# cap = None
# if (USE_WEBCAM == True):
#     cap = cv2.VideoCapture(0) # Webcam source
# else:
#     cap = cv2.VideoCapture('./demo/dinner.mp4') # Video file source

# # cap = cv2.VideoCapture('./demo/dinner.mp4') # Video file source

# while cap.isOpened():
#     ret, bgr_image = cap.read()
#     gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
#     rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)

#     faces = face_cascade.detectMultiScale(
#         gray_image, scaleFactor=1.1, minNeighbors=5,
#         minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE
#     )

#     current_time = time.time()
#     # Cập nhật cảm xúc mỗi 1 giây
#     if current_time - last_prediction_time >= prediction_interval:
#         last_prediction_time = current_time
#         latest_emotions = []  # reset danh sách cũ

#         for face_coordinates in faces:
#             x1, x2, y1, y2 = apply_offsets(face_coordinates, emotion_offsets)
#             gray_face = gray_image[y1:y2, x1:x2]
#             try:
#                 gray_face = cv2.resize(gray_face, (emotion_target_size))
#             except:
#                 continue

#             gray_face = preprocess_input(gray_face, True)
#             gray_face = np.expand_dims(gray_face, 0)
#             gray_face = np.expand_dims(gray_face, -1)

#             emotion_prediction = emotion_classifier.predict(gray_face)
#             emotion_probability = np.max(emotion_prediction)
#             emotion_label_arg = np.argmax(emotion_prediction)
#             emotion_text = emotion_labels[emotion_label_arg]

#             print(emotion_text, emotion_probability)

#             # Gán màu theo cảm xúc
#             if emotion_text == 'angry':
#                 color = emotion_probability * np.asarray((255, 0, 0))
#             elif emotion_text == 'sad':
#                 color = emotion_probability * np.asarray((0, 0, 255))
#             elif emotion_text == 'happy':
#                 color = emotion_probability * np.asarray((255, 255, 0))
#             elif emotion_text == 'surprise':
#                 color = emotion_probability * np.asarray((0, 255, 255))
#             else:
#                 color = emotion_probability * np.asarray((0, 255, 0))

#             color = color.astype(int).tolist()

#             # Lưu lại để vẽ mỗi frame
#             latest_emotions.append((face_coordinates, emotion_text, color))

#     # Vẽ lại mọi khuôn mặt với cảm xúc gần nhất (nếu có)
#     for i, face_coordinates in enumerate(faces):
#         if i < len(latest_emotions):
#             coords, emotion_text, color = latest_emotions[i]
#             draw_bounding_box(face_coordinates, rgb_image, color)
#             draw_text(face_coordinates, rgb_image, emotion_text,
#                       color, 0, -45, 1, 1)

#     bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
#     cv2.imshow('window_frame', bgr_image)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()

import cv2
import numpy as np
import time
from mtcnn.mtcnn import MTCNN
from keras.models import load_model
from statistics import mode
from utils.datasets import get_labels
from utils.inference import draw_text
from utils.inference import draw_bounding_box
from utils.inference import apply_offsets
from utils.preprocessor import preprocess_input

from datetime import datetime

# socketio
import socketio
sio = socketio.Client()
@sio.event
def connect():
    print("✅ Connected to Socket.IO server")
@sio.event
def disconnect():
    print("❌ Disconnected from Socket.IO server")
# Kết nối tới server
sio.connect("http://localhost:5000")

USE_WEBCAM = True # If false, loads video file source

# parameters for loading data and images
emotion_model_path = './models/emotion_model.hdf5'
emotion_labels = get_labels('fer2013')

# hyper-parameters for bounding boxes shape
frame_window = 10
emotion_offsets = (20, 40)

last_prediction_time = time.time()
prediction_interval = 2.0
latest_emotions = []

# loading models
emotion_classifier = load_model(emotion_model_path)

# getting input model shapes for inference
emotion_target_size = emotion_classifier.input_shape[1:3]

# MTCNN Face Detector initialization
detector = MTCNN()

# starting video streaming
cv2.namedWindow('window_frame')
video_capture = cv2.VideoCapture(0)

# Select video or webcam feed
cap = None
if (USE_WEBCAM == True):
    cap = cv2.VideoCapture(0) # Webcam source
else:
    cap = cv2.VideoCapture('./demo/dinner.mp4') # Video file source
# cap = cv2.VideoCapture('./demo/dinner.mp4') # Video file source

while cap.isOpened():
    ret, bgr_image = cap.read()
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)

    # MTCNN detects faces
    faces = detector.detect_faces(rgb_image)

    current_time = time.time()
    # Cập nhật cảm xúc mỗi 1 giây
    if current_time - last_prediction_time >= prediction_interval:
        last_prediction_time = current_time
        latest_emotions = []  # reset danh sách cũ

        for result in faces:  # faces là kết quả từ detector.detect_faces(rgb_image)
            x, y, width, height = result['box']
            x1 = max(x, 0)
            y1 = max(y, 0)
            x2 = x1 + width
            y2 = y1 + height

            face_coordinates = (x1, y1, width, height)
            x1_off, x2_off, y1_off, y2_off = apply_offsets((x1, y1, width, height), emotion_offsets)

            gray_face = gray_image[y1_off:y2_off, x1_off:x2_off]
            try:
                gray_face = cv2.resize(gray_face, (emotion_target_size))
            except:
                continue

            gray_face = preprocess_input(gray_face, True)
            gray_face = np.expand_dims(gray_face, 0)
            gray_face = np.expand_dims(gray_face, -1)

            emotion_prediction = emotion_classifier.predict(gray_face)
            emotion_probability = np.max(emotion_prediction)
            emotion_label_arg = np.argmax(emotion_prediction)
            emotion_text = emotion_labels[emotion_label_arg]

            print(emotion_text, emotion_probability)

            if emotion_text == 'angry':
                color = emotion_probability * np.asarray((255, 0, 0))
            elif emotion_text == 'sad':
                color = emotion_probability * np.asarray((0, 0, 255))
            elif emotion_text == 'happy':
                color = emotion_probability * np.asarray((255, 255, 0))
            elif emotion_text == 'surprise':
                color = emotion_probability * np.asarray((0, 255, 255))
            else:
                color = emotion_probability * np.asarray((0, 255, 0))

            color = color.astype(int).tolist()

            # Lưu lại để vẽ mỗi frame
            latest_emotions.append(((x1, y1, width, height), emotion_text, color))
            
            sio.emit('new_emotion', {
                'timestamp': str(datetime.now()),
                'emotion': emotion_text,
                'probability': float(emotion_probability),
                'color': color
            })


    # Vẽ lại mọi khuôn mặt với cảm xúc gần nhất (nếu có)
    for i, result in enumerate(faces):
        if i < len(latest_emotions):
            face_coordinates, emotion_text, color = latest_emotions[i]
            x, y, w, h = face_coordinates
            draw_bounding_box((x, y, w, h), rgb_image, color)
            draw_text((x, y, w, h), rgb_image, emotion_text, color, 0, -45, 1, 1)


    bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    cv2.imshow('window_frame', bgr_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
