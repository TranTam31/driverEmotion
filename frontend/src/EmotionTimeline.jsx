import React from 'react';

const EmotionTimeline = ({ emotions, trip, onSelectTime }) => {
  if (!emotions.length) {
    return (
      <div style={{ padding: '10px', backgroundColor: '#f9f9f9', borderRadius: '4px', textAlign: 'center' }}>
        No emotion data for this trip
      </div>
    );
  }

  const tripStart = new Date(trip.start_time).getTime();
  const tripEnd = trip.end_time ? new Date(trip.end_time).getTime() : Date.now();

  return (
    <div style={{ marginTop: '15px' }}>
      <h4>Emotion Timeline</h4>
      <div style={{
        position: 'relative', height: '40px',
        backgroundColor: '#f0f0f0', borderRadius: '4px',
        marginTop: '5px'
      }}>
        {emotions.map((emotion, index) => {
          const emotionTime = new Date(emotion.timestamp).getTime();
          const position = ((emotionTime - tripStart) / (tripEnd - tripStart)) * 100;
          const colorArray = emotion.color.split(',').map(Number);

          return (
            <div
              key={index}
              title={`${new Date(emotion.timestamp).toLocaleTimeString()} - ${emotion.emotion}`}
              onClick={() => onSelectTime(emotion.timestamp)}
              style={{
                position: 'absolute',
                left: `${position}%`,
                width: '4px',
                height: '100%',
                backgroundColor: `rgb(${colorArray})`,
                cursor: 'pointer'
              }}
            />
          );
        })}
      </div>
    </div>
  );
};

export default EmotionTimeline;
