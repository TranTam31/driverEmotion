import React from 'react';
import TimelineChart from './TimelineChart';
import VideoPlayer from './VideoPlayer';
import './App.css';
import Dashboard from './Dashboard';

function App() {
  return (
    <div className="App">
      <header className="App-header">
        <h1>Driver Emotion Monitoring System</h1>
      </header>
      <main>
        <Dashboard />
      </main>
    </div>
  );
}

export default App;