import React, { useEffect, useState } from 'react';
import axios from 'axios';
import { format } from 'date-fns';
import VideoPlayer from './VideoPlayer';
import TimelineChart from './TimelineChart';
import DriversList from './DriversList';
import TripPlayback from './TripPlayback';
import { toZonedTime } from 'date-fns-tz';
import AnalystView from './AnalystView';

const Dashboard = () => {
  const [drivers, setDrivers] = useState([]);
  const [selectedDriver, setSelectedDriver] = useState(null);
  const [viewMode, setViewMode] = useState('realtime'); // 'realtime' or 'playback'
  const [selectedDate, setSelectedDate] = useState(format(new Date(), 'yyyy-MM-dd'));
  const [driverTrips, setDriverTrips] = useState([]);
  const [selectedTrip, setSelectedTrip] = useState(null);
  const [flaggedDriverIds, setFlaggedDriverIds] = useState([]);

  // Fetch all drivers on component mount
  useEffect(() => {
    axios.get('http://localhost:5000/api/drivers')
      .then(response => {
        setDrivers(response.data);
      })
      .catch(error => console.error('Error fetching drivers:', error));

    axios.get('http://localhost:5000/api/flagged-emotions')
      .then(res => {
        const ids = [...new Set(res.data.map(item => item.driver_id))];
        setFlaggedDriverIds(ids);
      })
      .catch(err => console.error('Error loading flagged emotions:', err));
  }, []);

  // Fetch driver trips when driver or date changes
  useEffect(() => {
    if (selectedDriver && viewMode === 'playback') {
      axios.get(`http://localhost:5000/api/drivers/${selectedDriver.id}/trips`)
        .then(response => {
          // Filter trips by selected date
          const filteredTrips = response.data.filter(trip => {
            const tripDate = format(toZonedTime(new Date(trip.start_time), 'Etc/GMT'), 'yyyy-MM-dd');
            return tripDate === selectedDate;
          });
          setDriverTrips(filteredTrips);
          setSelectedTrip(null); // Reset selected trip when driver or date changes
        })
        .catch(error => {
          console.error('Error fetching driver trips:', error);
          setDriverTrips([]);
        });
    } else if (viewMode !== 'playback') {
      // Reset trips when not in playback mode
      setDriverTrips([]);
      setSelectedTrip(null);
    }
  }, [selectedDriver, selectedDate, viewMode]);

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

  return (
    <div className="dashboard-container" style={{ display: 'flex', height: '100vh', padding: '20px', maxWidth: '1200px', margin: '0 auto' }}>
      {/* Left Panel - Driver List (3/10 of screen width) */}
      <DriversList 
        drivers={drivers} 
        selectedDriver={selectedDriver} 
        onSelectDriver={setSelectedDriver} 
        flaggedDriverIds={flaggedDriverIds}
      />

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
              <button
                onClick={() => handleViewModeChange('analyst')}
                style={{
                  padding: '10px 20px',
                  backgroundColor: viewMode === 'analyst' ? '#f48a8a' : '#f1f1f1',
                  color: viewMode === 'analyst' ? 'white' : 'black',
                  border: 'none',
                  borderRadius: '4px',
                  cursor: 'pointer'
                }}
              >
                Analyst View
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
            ) : viewMode === 'playback' ? (
              /* Playback View */
              <TripPlayback 
                selectedDriver={selectedDriver}
                selectedDate={selectedDate}
                driverTrips={driverTrips}
                selectedTrip={selectedTrip}
                onDateChange={handleDateChange}
                onTripSelect={handleTripSelect}
              />
            ) : (
              <AnalystView driverId={selectedDriver.id} />
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