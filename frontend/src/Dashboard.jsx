import React, { useEffect, useState } from 'react';
import axios from 'axios';
import VideoPlayer from './VideoPlayer';
import TimelineChart from './TimelineChart';
import { format } from 'date-fns';

const Dashboard = () => {
  const [drivers, setDrivers] = useState([]);
  const [selectedDriver, setSelectedDriver] = useState(null);
  const [viewMode, setViewMode] = useState('realtime'); // 'realtime' or 'playback'
  const [selectedDate, setSelectedDate] = useState(format(new Date(), 'yyyy-MM-dd'));
  const [driverTrips, setDriverTrips] = useState([]);
  const [selectedTrip, setSelectedTrip] = useState(null);
  const [tripEmotions, setTripEmotions] = useState([]);
  const [selectedTimestamp, setSelectedTimestamp] = useState(null);
  const [videoUrl, setVideoUrl] = useState('');

  // Fetch all drivers on component mount
  useEffect(() => {
    axios.get('http://localhost:5000/api/drivers')
      .then(response => {
        setDrivers(response.data);
      })
      .catch(error => console.error('Error fetching drivers:', error));
  }, []);

  // Fetch driver trips when driver or date changes
  useEffect(() => {
    if (selectedDriver && viewMode === 'playback') {
      axios.get(`http://localhost:5000/api/drivers/${selectedDriver.id}/trips`)
        .then(response => {
          // Filter trips by selected date
          const filteredTrips = response.data.filter(trip => {
            const tripDate = new Date(trip.start_time).toISOString().split('T')[0];
            return tripDate === selectedDate;
          });
          setDriverTrips(filteredTrips);
          setSelectedTrip(null); // Reset selected trip when driver or date changes
        })
        .catch(error => console.error('Error fetching driver trips:', error));
    }
  }, [selectedDriver, selectedDate, viewMode]);

  // Fetch trip emotions when a trip is selected
  useEffect(() => {
    if (selectedTrip) {
      axios.get(`http://localhost:5000/api/trips/${selectedTrip.id}/emotions`)
        .then(response => {
          setTripEmotions(response.data);
          if (selectedTrip.video_path) {
            setVideoUrl(`../recordings/${selectedTrip.video_path}`);
          }
        })
        .catch(error => console.error('Error fetching trip emotions:', error));
    }
  }, [selectedTrip]);

  // Handle view mode change
  const handleViewModeChange = (mode) => {
    setViewMode(mode);
    setSelectedTrip(null); // Reset selected trip when changing view mode
  };

  // Handle date selection
  const handleDateChange = (event) => {
    setSelectedDate(event.target.value);
  };

  // Handle trip selection
  const handleTripSelect = (trip) => {
    setSelectedTrip(trip);
  };

  // Handle timeline selection
  const handleTimeSelect = (timestamp) => {
    setSelectedTimestamp(timestamp);
  };

  // Format trip time for display
  const formatTripTime = (timeString) => {
    if (!timeString) return 'N/A';
    const date = new Date(timeString);
    return format(date, 'HH:mm:ss');
  };

  return (
    <div className="dashboard-container" style={{ display: 'flex', height: '100vh' }}>
      {/* Left Panel - Driver List (3/10 of screen width) */}
      <div className="drivers-panel" style={{ width: '30%', padding: '20px', borderRight: '1px solid #ddd', overflow: 'auto' }}>
        <h2>Drivers</h2>
        <div className="drivers-list">
          {drivers.map(driver => (
            <div
              key={driver.id}
              className={`driver-item ${selectedDriver && selectedDriver.id === driver.id ? 'selected' : ''}`}
              onClick={() => setSelectedDriver(driver)}
              style={{
                padding: '10px',
                margin: '5px 0',
                border: '1px solid #ddd',
                borderRadius: '4px',
                cursor: 'pointer',
                backgroundColor: selectedDriver && selectedDriver.id === driver.id ? '#f0f8ff' : 'white'
              }}
            >
              <div className="driver-name" style={{ fontWeight: 'bold' }}>{driver.name}</div>
              <div className="driver-info" style={{ fontSize: '14px', color: '#666' }}>
                ID: {driver.id} | Status: {driver.status || 'Available'}
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Right Panel - Video and Timeline (7/10 of screen width) */}
      <div className="content-panel" style={{ width: '70%', padding: '20px' }}>
        {selectedDriver ? (
          <>
            {/* View Mode Selector */}
            <div className="view-mode-selector" style={{ marginBottom: '20px' }}>
              <button
                onClick={() => handleViewModeChange('realtime')}
                style={{
                  padding: '10px 20px',
                  marginRight: '10px',
                  backgroundColor: viewMode === 'realtime' ? '#4CAF50' : '#f1f1f1',
                  color: viewMode === 'realtime' ? 'white' : 'black',
                  border: 'none',
                  borderRadius: '4px',
                  cursor: 'pointer'
                }}
              >
                Realtime View
              </button>
              <button
                onClick={() => handleViewModeChange('playback')}
                style={{
                  padding: '10px 20px',
                  backgroundColor: viewMode === 'playback' ? '#2196F3' : '#f1f1f1',
                  color: viewMode === 'playback' ? 'white' : 'black',
                  border: 'none',
                  borderRadius: '4px',
                  cursor: 'pointer'
                }}
              >
                Playback View
              </button>
            </div>

            {/* Content based on selected view mode */}
            {viewMode === 'realtime' ? (
              /* Realtime View */
              <div className="realtime-view">
                <h3>Live Feed: {selectedDriver.name}</h3>
                <VideoPlayer driverId={selectedDriver.id}/>
                <div style={{ marginTop: '20px' }}>
                  <TimelineChart driverId={selectedDriver.id}/>
                </div>
              </div>
            ) : (
              /* Playback View */
              <div className="playback-view">
                <h3>Playback for: {selectedDriver.name}</h3>
                
                {/* Date Selector */}
                <div className="date-selector" style={{ marginBottom: '15px' }}>
                  <label htmlFor="trip-date" style={{ marginRight: '10px' }}>Select Date:</label>
                  <input
                    type="date"
                    id="trip-date"
                    value={selectedDate}
                    onChange={handleDateChange}
                    style={{ padding: '8px' }}
                  />
                </div>
                
                {/* Trips for Selected Date */}
                <div className="trips-container">
                  <h4>Trips on {selectedDate}</h4>
                  
                  {driverTrips.length > 0 ? (
                    <div className="trips-list" style={{ display: 'flex', flexDirection: 'column', gap: '10px', marginBottom: '20px' }}>
                      {driverTrips.map(trip => (
                        <div
                          key={trip.id}
                          className={`trip-item ${selectedTrip && selectedTrip.id === trip.id ? 'selected' : ''}`}
                          onClick={() => handleTripSelect(trip)}
                          style={{
                            padding: '10px',
                            border: '1px solid #ddd',
                            borderRadius: '4px',
                            cursor: 'pointer',
                            backgroundColor: selectedTrip && selectedTrip.id === trip.id ? '#e3f2fd' : 'white'
                          }}
                        >
                          <div style={{ marginBottom: '5px' }}>
                            <strong>Trip #{trip.id}</strong> | 
                            Start: {formatTripTime(trip.start_time)} | 
                            End: {formatTripTime(trip.end_time)}
                          </div>
                        </div>
                      ))}
                    </div>
                  ) : (
                    <div style={{ padding: '15px', backgroundColor: '#f9f9f9', borderRadius: '4px', textAlign: 'center' }}>
                      No trips found for this date
                    </div>
                  )}
                </div>
                
                {/* Selected Trip Playback */}
                {selectedTrip && (
                  <div className="selected-trip-playback">
                    <h4>Trip #{selectedTrip.id} Playback</h4>
                    
                    {/* Video Player for Playback */}
                    {selectedTrip.video_path ? (
                      <div className="video-playback" style={{ marginBottom: '15px' }}>
                        <video
                          width="640"
                          height="480"
                          controls
                          src={videoUrl}
                          style={{ border: '1px solid #ccc' }}
                        >
                        </video>
                      </div>
                    ) : (
                      <div style={{ width: '640px', height: '360px', display: 'flex', alignItems: 'center', justifyContent: 'center', backgroundColor: '#f0f0f0', border: '1px solid #ccc' }}>
                        No video available for this trip
                      </div>
                    )}
                    
                    {/* Timeline for the Selected Trip */}
                    {tripEmotions.length > 0 ? (
                      <div style={{ marginTop: '15px' }}>
                        <h4>Emotion Timeline</h4>
                        <div className="emotion-timeline" style={{ position: 'relative', height: '40px', backgroundColor: '#f0f0f0', borderRadius: '4px', marginTop: '5px' }}>
                          {tripEmotions.map((emotion, index) => {
                            // Calculate position based on timestamp within trip duration
                            const tripStart = new Date(selectedTrip.start_time).getTime();
                            const tripEnd = selectedTrip.end_time ? new Date(selectedTrip.end_time).getTime() : Date.now();
                            const emotionTime = new Date(emotion.timestamp).getTime();
                            const position = ((emotionTime - tripStart) / (tripEnd - tripStart)) * 100;
                            
                            // Parse color from string "r,g,b" to array
                            const colorArray = emotion.color.split(',').map(Number);
                            
                            return (
                              <div
                                key={index}
                                onClick={() => handleTimeSelect(emotion.timestamp)}
                                title={`${new Date(emotion.timestamp).toLocaleTimeString()} - ${emotion.emotion}`}
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
                    ) : (
                      <div style={{ padding: '10px', backgroundColor: '#f9f9f9', borderRadius: '4px', textAlign: 'center' }}>
                        No emotion data for this trip
                      </div>
                    )}
                  </div>
                )}
              </div>
            )}
          </>
        ) : (
          <div className="no-driver-selected" style={{ display: 'flex', height: '100%', alignItems: 'center', justifyContent: 'center', color: '#666' }}>
            <h3>Select a driver from the left panel</h3>
          </div>
        )}
      </div>
    </div>
  );
};

export default Dashboard;