import React from 'react';
import TimelineChart from './TimelineChart';
import VideoPlayer from './VideoPlayer';
import './App.css';

function App() {
  return (
    <div className="App">
      <h1>Emotion Detection Dashboard</h1>
      
      <div className="dashboard">
        <div className="video-section">
          <VideoPlayer />
        </div>
      </div>
      <div className="dashboard">
        <div className="timeline-section">
            <h2>Emotion Timeline</h2>
            <TimelineChart />
        </div>
      </div>
    </div>
  );
}

export default App;