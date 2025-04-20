# import cv2
# from datetime import datetime

# # Khởi tạo camera (0 là mặc định cho webcam, có thể thay đổi nếu sử dụng camera khác)
# cap = cv2.VideoCapture(0)

# # Kiểm tra nếu camera mở thành công
# if not cap.isOpened():
#     print("Không thể mở camera")
#     exit()

# # Lấy độ phân giải của video
# frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# # Định nghĩa codec và tên tệp video
# fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Định dạng codec MP4
# start_time = datetime.now()
# video_filename = f"session_{start_time.strftime('%Y%m%d_%H%M%S')}.mp4"
# fps = 24  # Số khung hình mỗi giây (fps)

# # Khởi tạo VideoWriter để ghi video vào tệp
# video_writer = cv2.VideoWriter(video_filename, fourcc, fps, (frame_width, frame_height))

# # Vòng lặp đọc và hiển thị từng khung hình
# while True:
#     ret, frame = cap.read()
#     if not ret:
#         print("Không thể nhận diện khung hình từ camera")
#         break

#     # Lấy thời gian hiện tại
#     current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

#     # Thêm thời gian vào khung hình
#     font = cv2.FONT_HERSHEY_SIMPLEX
#     cv2.putText(frame, current_time, (10, 30), font, 1, (255, 0, 0), 2, cv2.LINE_AA)

#     # Hiển thị video theo thời gian thực
#     cv2.imshow("Camera Realtime", frame)

#     # Ghi video vào tệp
#     video_writer.write(frame)

#     # Đợi phím 'q' để thoát khỏi vòng lặp
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # Giải phóng tài nguyên
# cap.release()
# video_writer.release()
# cv2.destroyAllWindows()

# print(f"Video đã được lưu với tên: {video_filename}")


# import cv2
# import threading
# import subprocess
# from datetime import datetime
# import time
# import os

# # Thông số
# duration = 10
# fps = 30
# width = 640
# height = 480
# start_time = datetime.now()
# timestamp_str = start_time.strftime('%Y%m%d_%H%M%S')

# video_filename = f"video_{timestamp_str}.avi"
# audio_filename = f"audio_{timestamp_str}.wav"
# output_filename = f"{timestamp_str}.mp4"

# # === Ghi âm thanh bằng ffmpeg ===
# def record_audio():
#     cmd = [
#         "ffmpeg",
#         "-y",
#         "-f", "dshow",
#         "-i", "audio=Microphone Array (Intel® Smart Sound Technology for Digital Microphones)",  # THAY bằng tên mic của bạn
#         "-t", str(duration),
#         audio_filename
#     ]
#     subprocess.run(cmd)

# # === Ghi hình bằng OpenCV (có timestamp) ===
# def record_video():
#     cap = cv2.VideoCapture(0)
#     cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
#     cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

#     fourcc = cv2.VideoWriter_fourcc(*'XVID')
#     out = cv2.VideoWriter(video_filename, fourcc, fps, (width, height))

#     print("🎥 Bắt đầu quay video có timestamp...")
#     start = time.time()

#     while time.time() - start < duration:
#         ret, frame = cap.read()
#         if not ret:
#             break

#         timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
#         cv2.putText(frame, timestamp, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
#                     1, (0, 255, 255), 2, cv2.LINE_AA)

#         cv2.imshow("Preview", frame)
#         out.write(frame)

#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     print("✅ Ghi hình xong.")
#     cap.release()
#     out.release()
#     cv2.destroyAllWindows()

# # === Chạy song song ghi hình + ghi âm ===
# audio_thread = threading.Thread(target=record_audio)
# video_thread = threading.Thread(target=record_video)

# video_thread.start()
# time.sleep(3.7)
# audio_thread.start()

# video_thread.join()
# audio_thread.join()

# # === Ghép âm thanh + video lại ===
# print("🔗 Ghép audio + video...")
# cmd_merge = [
#     "ffmpeg",
#     "-y",
#     "-i", video_filename,
#     "-i", audio_filename,
#     "-c:v", "libx264",
#     "-c:a", "aac",
#     "-strict", "experimental",
#     "-shortest",
#     output_filename
# ]
# subprocess.run(cmd_merge)

# print(f"🎉 Hoàn tất. File cuối: {output_filename}")

# # Cleanup (tuỳ chọn)
# os.remove(video_filename)
# os.remove(audio_filename)


import cv2
import subprocess
import threading
import time
from datetime import datetime
import os
import signal

# === Thông tin file ===
start_time = datetime.now()
timestamp_str = start_time.strftime('%Y%m%d_%H%M%S')
video_filename = f"video_{timestamp_str}.avi"
audio_filename = f"audio_{timestamp_str}.wav"
output_filename = f"{timestamp_str}.mp4"

# === Thông số video ===
fps = 30
width = 640
height = 480

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

# === Ghi hình có timestamp ===
def record_video():
    global recording
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(video_filename, fourcc, fps, (width, height))

    print("🎥 Bắt đầu ghi hình và hiển thị. Nhấn 'q' để dừng.")

    while recording:
        ret, frame = cap.read()
        if not ret:
            break

        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        cv2.putText(frame, timestamp, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow("Recording - Nhấn 'q' để dừng", frame)
        out.write(frame)

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
time.sleep(3.7)  # để webcam ổn định trước
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
