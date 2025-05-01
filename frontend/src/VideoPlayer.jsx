import React, { useEffect, useState, useRef } from 'react';
import { io } from 'socket.io-client';

const VideoPlayer = ({driverId}) => {
  const [hasStream, setHasStream] = useState(false);
  const canvasRef = useRef(null);
  const socketRef = useRef(null);
  const timeoutRef = useRef(null);
  
  console.log("Driver ID of video:", driverId);

  useEffect(() => {
    let socket = socketRef.current;
    
    if (!socket || socket.disconnected) {
      socket = io('http://localhost:5000');
      socketRef.current = socket;
    }

    // Xóa listeners cũ nếu có để tránh duplicate
    socket.off('connect');
    socket.off('video_frame');

    // Thiết lập listeners mới
    socket.on('connect', () => {
      console.log(`✅ Connected to video stream, joining room for driver ${driverId}`);
    });

    socket.on('video_frame', (data) => {
      if (data.driver_id !== driverId) {
        setHasStream(false);
        return;
      };

      setHasStream(true);
      // Clear any existing timeout
      if (timeoutRef.current) {
        clearTimeout(timeoutRef.current);
      }
      
      // Set timeout to mark stream as inactive after 3 seconds of no frames
      timeoutRef.current = setTimeout(() => {
        setHasStream(false);
      }, 3000);
      
      const canvas = canvasRef.current;
      if (!canvas) return;
      
      const ctx = canvas.getContext('2d');
      const img = new Image();
      img.onload = () => {
        ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
      };
      img.src = `data:image/jpeg;base64,${data.frame}`;
    });

    return () => {
      // Không đóng socket khi driverId thay đổi, chỉ xóa listeners
      socket.off('connect');
      socket.off('video_frame');
      
      if (timeoutRef.current) {
        clearTimeout(timeoutRef.current);
      }
    };
  }, [driverId]); // Thêm driverId vào dependency array

  // Thêm effect riêng để cleanup socket khi component unmount
  useEffect(() => {
    return () => {
      if (socketRef.current) {
        socketRef.current.disconnect();
      }
    };
  }, []);

  return (
    <div className="video-container">
      <h2>Live Emotion Detection</h2>
      
      {hasStream ? (
        <canvas 
          ref={canvasRef} 
          width="640" 
          height="480" 
          style={{ border: '1px solid #ccc' }}
        />
      ) : (
        <div 
          className="no-stream-message"
          style={{
            width: '640px',
            height: '480px',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            backgroundColor: '#f0f0f0',
            border: '1px solid #ccc',
            color: '#666',
            fontSize: '18px',
            fontWeight: 'bold'
          }}
        >
          No live stream available
        </div>
      )}
    </div>
  );
};

export default VideoPlayer;