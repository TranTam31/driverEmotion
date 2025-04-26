import cv2
import numpy as np
import time
import subprocess
import threading
import os
import mediapipe as mp
from tensorflow.keras.models import load_model
from statistics import mode
from utils.datasets import get_labels
from utils.inference import draw_text
from utils.inference import draw_bounding_box
from utils.inference import apply_offsets
from utils.preprocessor import preprocess_input
import tensorflow as tf
graph = tf.compat.v1.get_default_graph()

from models.cnn import MSEBlock
from models.cnn import ILABBlock

from datetime import datetime
import base64

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
emotion_model_path = './models/fer2013_mini_XCEPTION_final_acc_0.6620.keras'
emotion_labels = get_labels('fer2013')

# MediaPipe Face Detection initialization
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)

# hyper-parameters for bounding boxes shape
frame_window = 10
emotion_offsets = (20, 40)

prediction_interval = 2.0

# loading models
emotion_classifier = load_model(emotion_model_path, custom_objects={"MSEBlock": MSEBlock, "ILABBlock": ILABBlock})

# getting input model shapes for inference
emotion_target_size = emotion_classifier.input_shape[1:3]

# starting video streaming
start_time = datetime.now()
timestamp_str = start_time.strftime('%Y%m%d_%H%M%S')

video_filename = f"./recordings/video_{timestamp_str}.avi"
audio_filename = f"./recordings/audio_{timestamp_str}.wav"
output_filename = f"./recordings/{timestamp_str}.mp4"

# === Biến điều khiển ===
recording = True
ffmpeg_process = None  # để terminate sau

# === Ghi âm bằng ffmpeg (Popen để dừng được) ===
def record_audio():
    global ffmpeg_process
    cmd = [
        "ffmpeg",
        "-y",
        "-f", "dshow",
        "-i", "audio=Microphone Array (Intel® Smart Sound Technology for Digital Microphones)",  # Cập nhật theo máy bạn
        audio_filename
    ]
    ffmpeg_process = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def record_video():
    global recording
    
    fps = 30
    width = 640
    height = 480
    # starting video streaming
    cv2.namedWindow('window_frame')
    cap = cv2.VideoCapture(0)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(video_filename, fourcc, fps, (width, height))

    print(f"📹 Bắt đầu ghi video: {video_filename}")

    last_prediction_time = time.time()
    
    latest_emotions = []

    while recording:
        ret, bgr_image = cap.read()
        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
        
        # MediaPipe detects faces - image phải ở định dạng RGB
        results = face_detection.process(rgb_image)
        faces = []
        if results.detections:
            for detection in results.detections:
                bounding_box = detection.location_data.relative_bounding_box
                ih, iw, _ = rgb_image.shape
                x = int(bounding_box.xmin * iw)
                y = int(bounding_box.ymin * ih)
                width = int(bounding_box.width * iw)
                height = int(bounding_box.height * ih)
                faces.append({'box': (x, y, width, height)})

        current_time = time.time()
        # Cập nhật cảm xúc mỗi 1 giây
        if current_time - last_prediction_time >= prediction_interval:
            last_prediction_time = current_time
            latest_emotions = []  # reset danh sách cũ

            for result in faces:  # faces là kết quả đã được chuyển đổi từ MediaPipe
                x, y, width, height = result['box']
                x1 = max(x, 0)
                y1 = max(y, 0)
                x2 = x1 + width
                y2 = y1 + height

                face_coordinates = (x1, y1, width, height)
                x1_off, x2_off, y1_off, y2_off = apply_offsets((x1, y1, width, height), emotion_offsets)

                # Đảm bảo vị trí không vượt quá kích thước ảnh
                y1_off = max(0, y1_off)
                y2_off = min(gray_image.shape[0], y2_off)
                x1_off = max(0, x1_off)
                x2_off = min(gray_image.shape[1], x2_off)
                
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

        # Thêm timestamp vào từng frame (hiển thị trên ảnh - tuỳ chọn)
        timestamp_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        cv2.putText(bgr_image, timestamp_str, (10, height - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Add this code to stream the processed frame
        _, buffer = cv2.imencode('.jpg', bgr_image)
        jpg_as_text = base64.b64encode(buffer).decode('utf-8')
        
        # Emit the frame through Socket.IO
        sio.emit('video_frame', {
            'frame': jpg_as_text,
            'timestamp': str(datetime.now())
        })
        
        # Ghi frame đã xử lý vào file video
        cv2.imshow('window_frame', bgr_image)
        out.write(bgr_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            recording = False
            break

    print("🛑 Dừng ghi hình.")
    cap.release()
    out.release()
    cv2.destroyAllWindows()

# === Bắt đầu ghi âm và video song song ===
video_thread = threading.Thread(target=record_video)
audio_thread = threading.Thread(target=record_audio)

video_thread.start()
time.sleep(5)  # để webcam ổn định trước
audio_thread.start()

video_thread.join()

# === Dừng ffmpeg ghi âm nếu còn đang chạy ===
if ffmpeg_process and ffmpeg_process.poll() is None:
    print("🛑 Dừng ghi âm...")
    try:
        ffmpeg_process.terminate()  # Dùng terminate thay vì send_signal
        ffmpeg_process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        print("⚠️ Quá thời gian chờ. Buộc dừng ffmpeg.")
        ffmpeg_process.kill()

audio_thread.join()

# === Ghép audio + video ===
print("🔗 Ghép audio + video...")
cmd_merge = [
    "ffmpeg",
    "-y",
    "-i", video_filename,
    "-i", audio_filename,
    "-c:v", "libx264",
    "-c:a", "aac",
    "-strict", "experimental",
    "-shortest",
    output_filename
]
subprocess.run(cmd_merge)

print(f"✅ Hoàn tất! File cuối: {output_filename}")

# === (Tuỳ chọn) Xoá file tạm ===
os.remove(video_filename)
os.remove(audio_filename)