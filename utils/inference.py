import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

def load_image(image_path, grayscale=False, target_size=None):
    """
    Tải hình ảnh từ đường dẫn
    """
    if grayscale:
        img = Image.open(image_path).convert('L')
    else:
        img = Image.open(image_path).convert('RGB')
    
    if target_size:
        img = img.resize(target_size)
    
    return np.array(img)

def load_detection_model(model_path):
    """
    Tải mô hình phát hiện khuôn mặt Haar Cascade
    """
    detection_model = cv2.CascadeClassifier(model_path)
    return detection_model

def detect_faces(detection_model, gray_image_array):
    """
    Phát hiện khuôn mặt từ ảnh xám
    """
    return detection_model.detectMultiScale(gray_image_array, 1.3, 5)

def draw_bounding_box(face_coordinates, image_array, color):
    """
    Vẽ hộp giới hạn xung quanh khuôn mặt
    """
    x, y, w, h = face_coordinates
    cv2.rectangle(image_array, (x, y), (x + w, y + h), color, 2)

def apply_offsets(face_coordinates, offsets):
    """
    Áp dụng offset cho tọa độ khuôn mặt
    """
    x, y, width, height = face_coordinates
    x_off, y_off = offsets
    return (x - x_off, x + width + x_off, y - y_off, y + height + y_off)

def draw_text(coordinates, image_array, text, color, x_offset=0, y_offset=0,
                                              font_scale=2, thickness=2):
    """
    Vẽ văn bản lên ảnh
    """
    x, y = coordinates[:2]
    cv2.putText(image_array, text, (x + x_offset, y + y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, color, thickness, cv2.LINE_AA)

def get_colors(num_classes):
    """
    Tạo màu sắc cho mỗi lớp dựa trên biểu đồ màu hsv
    """
    colors = plt.cm.hsv(np.linspace(0, 1, num_classes)).tolist()
    colors = np.asarray(colors) * 255
    return colors