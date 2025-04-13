// import React, { useEffect, useState } from 'react';
// import axios from 'axios';

// const TimelineChart = () => {
//   const [emotions, setEmotions] = useState([]);

//   useEffect(() => {
//     axios.get('http://localhost:5000/api/emotions')
//       .then(response => setEmotions(response.data))
//       .catch(error => console.error(error));
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

// import React, { useEffect, useState } from 'react';
// import axios from 'axios';
// import { io } from 'socket.io-client';

// const TimelineChart = () => {
//   const [emotions, setEmotions] = useState([]);

//   useEffect(() => {
//     // Fetch dữ liệu ban đầu từ API
//     axios.get('http://localhost:5000/api/emotions')
//       .then(response => setEmotions(response.data))
//       .catch(error => console.error(error));

//     // Kết nối WebSocket
//     const socket = io("http://localhost:5000");

//     socket.on('connect', () => {
//         console.log('Socket.IO connected');
//       });
  
//       socket.on('disconnect', () => {
//         console.log('Socket.IO disconnected');
//       });
  
//     // Lắng nghe sự kiện 'new_emotion'
//     socket.on("new_emotion", (newEmotion) => {
//       setEmotions(prev => [...prev, newEmotion]);
//       console.log("New emotion received:", newEmotion);
//     });

//     // Ngắt kết nối khi component bị huỷ
//     return () => {
//       socket.disconnect();
//     };
//   }, []);

//   return (
//     <div className="timeline" style={{ padding: '20px', whiteSpace: 'nowrap', overflowX: 'auto' }}>
//       {emotions.map((e, idx) => (
//         <div key={idx}
//           title={`${e.timestamp} - ${e.emotion}`}
//           style={{
//             backgroundColor: `rgb(${e.color})`,
//             width: '5px',
//             height: '50px',
//             display: 'inline-block',
//             marginRight: '1px',
//             borderRadius: '2px'
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

const TimelineChart = () => {
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

  return (
    <div className="timeline">
      {emotions.map((e, idx) => (
        <div key={idx}
          title={`${e.timestamp} - ${e.emotion}`}
          style={{
            backgroundColor: `rgb(${e.color})`,
            width: '5px',
            height: '50px',
            display: 'inline-block',
            marginRight: '1px'
          }}>
        </div>
      ))}
    </div>
  );
};

export default TimelineChart;
