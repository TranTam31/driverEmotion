import cv2
import numpy as np
from PIL import Image

def preprocess_input(x, v2=True):
    """
    Xử lý dữ liệu đầu vào trước khi đưa vào mô hình
    """
    x = x.astype('float32')
    x = x / 255.0
    if v2:
        x = x - 0.5
        x = x * 2.0
    return x

def _imread(image_name):
    """
    Thay thế hàm imread từ scipy.misc
    """
    img = cv2.imread(image_name)
    if img is not None:
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        # Thử với PIL nếu cv2 không đọc được
        return np.array(Image.open(image_name))

def _imresize(image_array, size):
    """
    Thay thế hàm imresize từ scipy.misc
    """
    if isinstance(size, tuple):
        height, width = size
    elif isinstance(size, (int, float)):
        # Nếu size là một số, giữ nguyên tỷ lệ ảnh
        height, width = size, size
    else:
        raise ValueError("size phải là tuple hoặc số")
    
    return cv2.resize(image_array, (width, height))

def to_categorical(integer_classes, num_classes=2):
    """
    Chuyển đổi nhãn số nguyên thành dạng one-hot encoding
    """
    integer_classes = np.asarray(integer_classes, dtype='int')
    num_samples = integer_classes.shape[0]
    categorical = np.zeros((num_samples, num_classes))
    categorical[np.arange(num_samples), integer_classes] = 1
    return categorical