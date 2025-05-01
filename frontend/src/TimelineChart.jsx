import React, { useEffect, useState, useRef } from 'react';
import { io } from 'socket.io-client';

const TimelineChart = ({driverId}) => {
  const [emotions, setEmotions] = useState([]);
  const [live, setLive] = useState(true);
  const timelineRef = useRef(null);
  const socketRef = useRef(null);

  // Connect to WebSocket for realtime updates
  useEffect(() => {
    const socket = io('http://localhost:5000');
    socketRef.current = socket;

    socket.on('connect', () => {
      console.log('✅ Connected to Socket.IO server for emotions');
    });

    socket.on('new_emotion', (data) => {
      if (data.driver_id !== driverId) {
        setLive(false); // Set live to false if not the same driver
        return; // Ignore if not the same driver
      }
      
      setLive(true); // Set live to true if the same driver
      console.log('🎉 New emotion received:', data);
      setEmotions(prev => [...prev, data]);

      if (timelineRef.current) {
        timelineRef.current.scrollLeft = timelineRef.current.scrollWidth;
      }
    });

    return () => {
      socket.disconnect();
    };
  }, [driverId]);

  useEffect(() => {
      return () => {
        if (socketRef.current) {
          socketRef.current.disconnect();
        }
      };
    }, []);

  // Create emotion bars for timeline
  const renderEmotionBars = () => {
    if (live === false) {
      return <div className="no-data">Chưa có dữ liệu cảm xúc</div>;
    }

    return emotions.map((e, idx) => (
      <div 
        key={idx}
        title={`${e.timestamp} - ${e.emotion}`}
        style={{
          backgroundColor: `rgb(${e.color})`,
          width: '4px',  // Giảm xuống còn 4px
          height: '30px', // Giảm chiều cao xuống 30px
          display: 'inline-block',
          cursor: 'pointer' // Thêm con trỏ để cho thấy có thể click
        }}
      />
    ));
  };

  return (
    <div>
      {/* Tiêu đề cho timeline */}
      <div style={{ marginBottom: '5px', fontSize: '14px', fontWeight: 'bold' }}>
        Realtime Emotion Timeline
      </div>
      
      {/* Container cho timeline */}
      <div 
        className="timeline-container" 
        style={{
          backgroundColor: '#f0f0f0',
          padding: '5px',
          borderRadius: '4px',
          position: 'relative'
        }}
      >
        <div 
          ref={timelineRef}
          className="timeline" 
          style={{
            display: 'flex',
            height: '30px',
            alignItems: 'center',
            overflowX: 'auto',
            minWidth: '100%',
            position: 'relative',
            whiteSpace: 'nowrap'
          }}
        >
          {renderEmotionBars()}
        </div>
      </div>
    </div>
  );
};

export default TimelineChart;