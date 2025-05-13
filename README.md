# Tracking Emotion Driver Webapp

## 1. Trước khi chạy, vào MySQL workbench tạo database
- Tạo schema có tên là:

  ```
  emotion_db
  ```
- Chạy đoạn sau:
  ```sql
  USE emotion_db
  -- Create drivers table
  CREATE TABLE IF NOT EXISTS drivers (
      id INT AUTO_INCREMENT PRIMARY KEY,
      name VARCHAR(100) NOT NULL,
      license_number VARCHAR(50),
      phone VARCHAR(20),
      created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
  );
  
  -- Create trips table
  CREATE TABLE IF NOT EXISTS trips (
      id INT AUTO_INCREMENT PRIMARY KEY,
      driver_id INT NOT NULL,
      start_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
      end_time TIMESTAMP NULL,
      video_path VARCHAR(255),
      status ENUM('active', 'completed', 'cancelled') DEFAULT 'active',
      FOREIGN KEY (driver_id) REFERENCES drivers(id)
  );
  
  -- Create emotion_log table
  CREATE TABLE IF NOT EXISTS emotion_log (
      id INT AUTO_INCREMENT PRIMARY KEY,
      timestamp DATETIME NOT NULL,
      emotion VARCHAR(50),
      probability FLOAT,
      color VARCHAR(20),
      trip_id INT,
      is_check BOOLEAN DEFAULT FALSE,
      FOREIGN KEY (trip_id) REFERENCES trips(id)
  );
  
  -- Insert some sample drivers for testing
  INSERT INTO drivers (name, license_number, phone) VALUES 
  ('Nguyen Van A', 'DL001', '0901234567'),
  ('Tran Thi B', 'DL002', '0909876543'),
  ('Le Van C', 'DL003', '0905555555'),
  ('Pham Thi D', 'DL004', '0908888888');
  ```

## 2. Chạy chương trình
- Cài thư viện:
  ```
  pip install -r requirements.txt
  ```
- Chạy server:
  ```
  cd backend
  python app.py
  ```
- Chạy frontend:
  ```
  cd frontend
  npm i
  npm start
  ```
- Chạy file lấy emotion: 

  ```
  python emotions.py
  ```
