from flask import Flask, jsonify, request
from flask_cors import CORS
from flask_socketio import SocketIO
import mysql.connector

app = Flask(__name__)
CORS(app, origins="http://localhost:3000")
socketio = SocketIO(app, cors_allowed_origins="*")

# Hàm kết nối MySQL
def get_db_connection():
    return mysql.connector.connect(
        host="localhost", user="root", password="123456", database="emotion_db"
    )

# API lấy tất cả cảm xúc
@app.route('/api/emotions')
def get_emotions():
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT * FROM emotion_log ORDER BY timestamp ASC")
    results = cursor.fetchall()
    cursor.close()
    conn.close()
    return jsonify(results)

#BẮT SỰ KIỆN KẾT NỐI
@socketio.on('connect')
def handle_connect():
    print('Client connected')

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')

@socketio.on('new_emotion')
def handle_new_emotion(data):
    timestamp = data.get('timestamp')
    emotion = data.get('emotion')
    probability = data.get('probability')
    color = data.get('color')

    print("📥 Received emotion from client:", emotion)

    # Lưu vào DB
    # conn = get_db_connection()
    # cursor = conn.cursor()
    # color_str = ','.join(map(str, color))
    # cursor.execute("INSERT INTO emotion_log (timestamp, emotion, probability, color) VALUES (%s, %s, %s, %s)",
    #                (timestamp, emotion, float(probability), color_str))
    # conn.commit()
    # cursor.close()
    # conn.close()

    # Phát lại cho tất cả frontend đã kết nối
    socketio.emit('new_emotion', data)


# Khởi động app
if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000)