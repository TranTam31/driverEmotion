from flask import Flask, jsonify, request
from flask_cors import CORS
from flask_socketio import SocketIO
import mysql.connector
from datetime import datetime
import datetime
import jwt
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
CORS(app, origins="http://localhost:3000")
socketio = SocketIO(app, cors_allowed_origins="*")

app.config['SECRET_KEY'] = 'your_secret_key'

# Hàm kết nối MySQL
def get_db_connection():
    return mysql.connector.connect(
        host="localhost", user="root", password="123456", database="emotion_db"
    )

# API lấy tất cả tài xế
@app.route('/api/drivers')
def get_drivers():
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT * FROM drivers ORDER BY name ASC")
    results = cursor.fetchall()
    cursor.close()
    conn.close()
    return jsonify(results)

# API tạo chuyến đi mới
@app.route('/api/trips', methods=['POST'])
def create_trip():
    data = request.json
    driver_id = data.get('driver_id')
    
    if not driver_id:
        return jsonify({"error": "Driver ID is required"}), 400
    
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    
    # Tạo chuyến đi mới
    cursor.execute(
        "INSERT INTO trips (driver_id, start_time, status) VALUES (%s, %s, %s)",
        (driver_id, datetime.now(), 'active')
    )
    conn.commit()
    
    # Lấy ID của chuyến đi vừa tạo
    trip_id = cursor.lastrowid
    
    # Lấy thông tin của chuyến đi
    cursor.execute("SELECT * FROM trips WHERE id = %s", (trip_id,))
    trip = cursor.fetchone()
    
    cursor.close()
    conn.close()
    
    return jsonify(trip)

# API cập nhật đường dẫn video cho chuyến đi
@app.route('/api/trips/<int:trip_id>', methods=['PUT'])
def update_trip(trip_id):
    data = request.json
    video_path = data.get('video_path')
    status = data.get('status', 'active')
    
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    
    update_fields = []
    update_values = []
    
    if video_path:
        update_fields.append("video_path = %s")
        update_values.append(video_path)
    
    if status == 'completed':
        update_fields.append("end_time = %s")
        update_values.append(datetime.now())
        update_fields.append("status = %s")
        update_values.append('completed')
    
    if update_fields:
        cursor.execute(
            f"UPDATE trips SET {', '.join(update_fields)} WHERE id = %s",
            (*update_values, trip_id)
        )
        conn.commit()
    
    # Lấy thông tin đã cập nhật
    cursor.execute("SELECT * FROM trips WHERE id = %s", (trip_id,))
    trip = cursor.fetchone()
    
    cursor.close()
    conn.close()
    
    return jsonify(trip)

# API lấy tất cả cảm xúc theo chuyến đi
@app.route('/api/trips/<int:trip_id>/emotions')
def get_trip_emotions(trip_id):
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT * FROM emotion_log WHERE trip_id = %s ORDER BY timestamp ASC", (trip_id,))
    results = cursor.fetchall()
    cursor.close()
    conn.close()
    return jsonify(results)

# API lấy tất cả chuyến đi theo tài xế
@app.route('/api/drivers/<int:driver_id>/trips')
def get_driver_trips(driver_id):
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT * FROM trips WHERE driver_id = %s ORDER BY start_time DESC", (driver_id,))
    results = cursor.fetchall()
    cursor.close()
    conn.close()
    return jsonify(results)

# API lấy tất cả cảm xúc (giữ lại để tương thích với code cũ)
@app.route('/api/emotions')
def get_emotions():
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT * FROM emotion_log ORDER BY timestamp ASC")
    results = cursor.fetchall()
    cursor.close()
    conn.close()
    return jsonify(results)

# NEW API: Lấy chuyến đi theo ngày
@app.route('/api/trips/by-date/<string:date>')
def get_trips_by_date(date):
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    
    # Format date for SQL query (assumes date is in YYYY-MM-DD format)
    date_start = f"{date} 00:00:00"
    date_end = f"{date} 23:59:59"
    
    cursor.execute(
        "SELECT * FROM trips WHERE start_time BETWEEN %s AND %s ORDER BY start_time ASC",
        (date_start, date_end)
    )
    results = cursor.fetchall()
    cursor.close()
    conn.close()
    return jsonify(results)

# NEW API: Lấy chuyến đi theo ngày và tài xế
@app.route('/api/drivers/<int:driver_id>/trips/by-date/<string:date>')
def get_driver_trips_by_date(driver_id, date):
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    
    # Format date for SQL query
    date_start = f"{date} 00:00:00"
    date_end = f"{date} 23:59:59"
    
    cursor.execute(
        "SELECT * FROM trips WHERE driver_id = %s AND start_time BETWEEN %s AND %s ORDER BY start_time ASC",
        (driver_id, date_start, date_end)
    )
    results = cursor.fetchall()
    cursor.close()
    conn.close()
    return jsonify(results)

# NEW API: Lấy thông tin cảm xúc tại một thời điểm cụ thể
@app.route('/api/trips/<int:trip_id>/emotions/<string:timestamp>')
def get_emotion_at_timestamp(trip_id, timestamp):
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    
    # Find the emotion closest to the given timestamp
    cursor.execute(
        """
        SELECT * FROM emotion_log 
        WHERE trip_id = %s 
        ORDER BY ABS(TIMESTAMPDIFF(SECOND, timestamp, %s)) 
        LIMIT 1
        """,
        (trip_id, timestamp)
    )
    result = cursor.fetchone()
    cursor.close()
    conn.close()
    
    if result:
        return jsonify(result)
    else:
        return jsonify({"error": "No emotion data found at the specified timestamp"}), 404

# New API
@app.route('/api/flagged-emotions')
def get_flagged_emotions():
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)

    query = """
        SELECT 
            d.id AS driver_id,
            d.name AS driver_name,
            t.id AS trip_id,
            e.id,
            e.emotion,
            e.timestamp,
            e.is_check
        FROM drivers d
        JOIN trips t ON d.id = t.driver_id
        JOIN emotion_log e ON t.id = e.trip_id
        WHERE e.emotion IN ('angry', 'sad', 'surprise') AND e.is_check = FALSE
        ORDER BY e.timestamp DESC
    """

    cursor.execute(query)
    results = cursor.fetchall()

    cursor.close()
    conn.close()

    return jsonify(results)

#new api
@app.route('/api/emotions/<int:emotion_id>/check', methods=['POST'])
def mark_emotion_checked(emotion_id):
    conn = get_db_connection()
    cursor = conn.cursor()

    # Cập nhật cột is_check thành TRUE
    cursor.execute("UPDATE emotion_log SET is_check = TRUE WHERE id = %s", (emotion_id,))
    conn.commit()
    
    affected_rows = cursor.rowcount
    cursor.close()
    conn.close()

    if affected_rows == 0:
        return jsonify({'success': False, 'message': 'Emotion not found'}), 404
    return jsonify({'success': True, 'message': 'Emotion marked as checked'})

@app.route('/api/register', methods=['POST'])
def register():
    data = request.get_json()
    email = data['email']
    password = generate_password_hash(data['password'])

    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("INSERT INTO users (email, password) VALUES (%s, %s)", (email, password))
    conn.commit()
    cursor.close()
    conn.close()
    return jsonify({'message': 'User registered successfully'}), 201

@app.route('/api/login', methods=['POST'])
def login():
    data = request.get_json()
    email = data['email']
    password = data['password']

    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT * FROM users WHERE email=%s", (email,))
    user = cursor.fetchone()
    cursor.close()
    conn.close()

    if user and check_password_hash(user['password'], password):
        token = jwt.encode({
            'email': user['email'],
            'exp': datetime.datetime.utcnow() + datetime.timedelta(hours=1)
        }, app.config['SECRET_KEY'], algorithm="HS256")
        return jsonify({'token': token})
    else:
        return jsonify({'message': 'Invalid credentials'}), 401

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
    trip_id = data.get('trip_id')  # Thêm trip_id
    driver_id = data.get('driver_id')

    print(f"📥 Received emotion from client: {emotion} (Trip ID: {trip_id})")

    # Lưu vào DB
    conn = get_db_connection()
    cursor = conn.cursor()
    color_str = ','.join(map(str, color))
    cursor.execute(
        "INSERT INTO emotion_log (timestamp, emotion, probability, color, trip_id) VALUES (%s, %s, %s, %s, %s)",
        (timestamp, emotion, float(probability), color_str, trip_id)
    )
    conn.commit()
    cursor.close()
    conn.close()

    if driver_id:
        # Phát lại cho tất cả frontend đã kết nối
        socketio.emit('new_emotion', data)

@socketio.on('video_frame')
def handle_video_frame(data):
    driver_id = data.get('driver_id')
    if driver_id:
        # Simply pass the frame to all connected clients
        socketio.emit('video_frame', data)

# Khởi động app
if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000)