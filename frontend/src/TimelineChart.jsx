// import React, { useEffect, useState } from 'react';
// import axios from 'axios';
// import { io } from 'socket.io-client';

// const TimelineChart = () => {
//   const [emotions, setEmotions] = useState([]);

//   // Fetch dữ liệu ban đầu
//   useEffect(() => {
//     axios.get('http://localhost:5000/api/emotions')
//       .then(response => setEmotions(response.data))
//       .catch(error => console.error(error));
//   }, []);

//   // Kết nối WebSocket
//   useEffect(() => {
//     const socket = io('http://localhost:5000');

//     socket.on('connect', () => {
//       console.log('✅ Connected to Socket.IO server');
//     });

//     socket.on('new_emotion', (data) => {
//       console.log('🎉 New emotion received:', data);
//       setEmotions(prev => [...prev, data]); // Thêm emotion mới vào cuối
//     });

//     return () => {
//       socket.disconnect();
//     };
//   }, []);

//   return (
//     <div className="timeline">
//       {emotions.map((e, idx) => (
//         <div key={idx}
//           title={`${e.timestamp} - ${e.emotion}`}
//           style={{
//             backgroundColor: `rgb(${e.color})`,
//             width: '5px',
//             height: '50px',
//             display: 'inline-block',
//             marginRight: '1px'
//           }}>
//         </div>
//       ))}
//     </div>
//   );
// };

// export default TimelineChart;

import React, { useEffect, useState } from 'react';
import axios from 'axios';
import { io } from 'socket.io-client';

const TimelineChart = ({ onTimeSelect }) => {
  const [emotions, setEmotions] = useState([]);

  // Fetch dữ liệu ban đầu
  useEffect(() => {
    axios.get('http://localhost:5000/api/emotions')
      .then(response => setEmotions(response.data))
      .catch(error => console.error(error));
  }, []);

  // Kết nối WebSocket
  useEffect(() => {
    const socket = io('http://localhost:5000');

    socket.on('connect', () => {
      console.log('✅ Connected to Socket.IO server');
    });

    socket.on('new_emotion', (data) => {
      console.log('🎉 New emotion received:', data);
      setEmotions(prev => [...prev, data]); // Thêm emotion mới vào cuối
    });

    return () => {
      socket.disconnect();
    };
  }, []);

  // Hàm xử lý khi click vào thanh timeline
  const handleTimelineClick = (e, timestamp) => {
    // Gọi hàm callback để thông báo cho component cha biết thời điểm được chọn
    if (onTimeSelect) {
      onTimeSelect(timestamp);
    }
    console.log('Đã chọn thời điểm:', timestamp);
  };

  // Tạo các phần tử emotion cho timeline
  const renderEmotionBars = () => {
    if (emotions.length === 0) {
      return <div className="no-data">Chưa có dữ liệu cảm xúc</div>;
    }

    return emotions.map((e, idx) => (
      <div 
        key={idx}
        onClick={(event) => handleTimelineClick(event, e.timestamp)}
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
        Timeline Video (click để xem lại theo thời gian)
      </div>
      
      {/* Container cho timeline */}
      <div 
        className="timeline-container" 
        style={{
          backgroundColor: '#f0f0f0',
          padding: '5px',
          borderRadius: '4px',
          overflowX: 'auto'
        }}
      >
        <div 
          className="timeline" 
          style={{
            display: 'flex',
            height: '30px',
            alignItems: 'center',
            minWidth: '100%',
            position: 'relative'
          }}
        >
          {renderEmotionBars()}
        </div>
      </div>
    </div>
  );
};

export default TimelineChart;