import React, { useEffect, useState } from 'react';
import axios from 'axios';
import { format } from 'date-fns';
import { toZonedTime } from 'date-fns-tz';

const TripPlayback = ({ 
  selectedDriver, 
  selectedDate, 
  driverTrips = [], 
  selectedTrip, 
  onDateChange, 
  onTripSelect 
}) => {
  const [tripEmotions, setTripEmotions] = useState([]);
  const [flaggedEmotions, setFlaggedEmotions] = useState([]);
  const [showFlaggedTable, setShowFlaggedTable] = useState(false);
  const [videoUrl, setVideoUrl] = useState('');

  // Fetch emotions flagged for the selected driver
  useEffect(() => {
    if (selectedDriver?.id) {
      axios.get('http://localhost:5000/api/flagged-emotions')
        .then(response => {
          const filtered = response.data.filter(e => e.driver_id === selectedDriver.id);
          setFlaggedEmotions(filtered);
        })
        .catch(error => console.error('Error fetching flagged emotions:', error));
    }
  }, [selectedDriver]);

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

  // Handle marking an emotion as checked
  const handleCheckEmotion = (emotionId) => {
    const confirmCheck = window.confirm("Do you sure that this emotion is normal?");
    
    // Chỉ tiếp tục nếu người dùng nhấn OK
    if (confirmCheck) {
      axios.post(`http://localhost:5000/api/emotions/${emotionId}/check`)
        .then(response => {
          setFlaggedEmotions(prev => prev.map(e => e.id === emotionId ? { ...e, is_check: true } : e));
          // Có thể thêm thông báo thành công nếu cần
          alert("Đã đánh dấu thành công!");
        })
        .catch(error => {
          console.error('Error checking emotion:', error);
          // Thông báo lỗi cho người dùng
          alert("Có lỗi xảy ra khi đánh dấu emotion!");
        });
    }
    // Nếu người dùng nhấn Cancel thì không làm gì cả
  };

  // Xử lý chọn ngày từ timestamp trong bảng cảm xúc
  const handleSelectDateFromTimestamp = (timestamp) => {
    // Chuyển timestamp thành đối tượng Date
    const date = new Date(timestamp);
    // Format thành chuỗi YYYY-MM-DD để sử dụng với input type="date"
    const formattedDate = format(toZonedTime(new Date(date), 'Etc/GMT'), 'yyyy-MM-dd');
    // Gọi hàm onDateChange với đối tượng event giả
    onDateChange({ target: { value: formattedDate } });
  };

  const formatTripTime = (timeString) => {
    if (!timeString) return 'N/A';
    const date = new Date(timeString);
    return format(toZonedTime(new Date(date), 'Etc/GMT'), 'HH:mm:ss')
  };

  return (
    <div className="playback-view">
      <h3>Playback for: {selectedDriver.name}</h3>

      {/* Flagged Emotions Table */}
      {flaggedEmotions.length > 0 && (
        <div className="flagged-emotions-table" style={{ marginBottom: '20px' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '10px' }}>
            <h4 style={{ margin: 0 }}>Negative Emotions</h4>
            <div>
              <button
                onClick={() => setShowFlaggedTable(prev => !prev)}
                style={{
                  padding: '8px 16px',
                  backgroundColor: '#1976d2',
                  color: 'white',
                  border: 'none',
                  borderRadius: '6px',
                  cursor: 'pointer'
                }}
              >
                {showFlaggedTable ? 'Hide details' : 'Show details'}
              </button>
            </div>
          </div>
          {showFlaggedTable && (<table style={{ width: '100%', borderCollapse: 'collapse' }}>
            <thead>
              <tr style={{ backgroundColor: '#eee' }}>
                <th style={{ padding: '8px', border: '1px solid #ccc' }}>Timestamp</th>
                <th style={{ padding: '8px', border: '1px solid #ccc' }}>Emotion</th>
                <th style={{ padding: '8px', border: '1px solid #ccc' }}>Trip ID</th>
                <th style={{ padding: '8px', border: '1px solid #ccc' }}>Action</th>
                <th style={{ padding: '8px', border: '1px solid #ccc' }}>Check</th>
              </tr>
            </thead>
            <tbody style={{ textAlign: 'center' }}>
              {flaggedEmotions.map(e => (
                <tr key={e.id}>
                  <td style={{ padding: '8px', border: '1px solid #ddd' }}>{toZonedTime(new Date(e.timestamp), 'Etc/GMT').toLocaleString('en-GB', { hour12: false })}</td>
                  <td style={{ padding: '8px', border: '1px solid #ddd' }}>{e.emotion}</td>
                  <td style={{ padding: '8px', border: '1px solid #ddd' }}>{e.trip_id}</td>
                  <td style={{ padding: '8px', border: '1px solid #ddd' }}>
                    <button
                      onClick={() => handleSelectDateFromTimestamp(e.timestamp)}
                      style={{ 
                        padding: '4px 10px', 
                        backgroundColor: '#ff9800', 
                        color: 'white', 
                        border: 'none', 
                        borderRadius: '4px', 
                        cursor: 'pointer',
                        fontSize: '0.8rem'
                      }}
                      title="Select this date"
                    >
                      Select Date
                    </button>
                  </td>
                  <td style={{ padding: '8px', border: '1px solid #ddd' }}>
                    {e.is_check ? 'Checked' : (
                      <button
                        onClick={() => handleCheckEmotion(e.id)}
                        style={{ padding: '4px 10px', backgroundColor: '#4caf50', color: 'white', border: 'none', borderRadius: '4px', cursor: 'pointer', fontSize: '0.8rem' }}
                      >
                        Normal
                      </button>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>)}
        </div>
      )}

      <div 
        className="divider" 
        style={{ 
          borderTop: '3px solid #ddd', 
          margin: '15px 0', 
          width: '100%' 
        }}
      ></div>

      {/* Date Selector */}
      <div className="date-selector" style={{ marginBottom: '15px' }}>
        <label htmlFor="trip-date" style={{ marginRight: '10px' }}><b>Select Date:</b></label>
        <input
          type="date"
          id="trip-date"
          value={selectedDate}
          onChange={onDateChange}
          style={{ padding: '8px', fontFamily: 'Arial, sans-serif' }}
        />
      </div>

      {/* Trips for Selected Date */}
      <div className="trips-container">
        {driverTrips.length > 0 ? (
          <div className="trips-list" style={{ display: 'flex', flexDirection: 'column', gap: '10px', marginBottom: '20px' }}>
            {driverTrips && driverTrips.map(trip => (
              <div
                key={trip.id}
                className={`trip-item ${selectedTrip && selectedTrip.id === trip.id ? 'selected' : ''}`}
                onClick={() => onTripSelect(trip)}
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
        <div className="selected-trip-playback" style={{paddingBottom: '8px'}}>
          <h4>Trip #{selectedTrip.id} Playback</h4>
          {selectedTrip.video_path ? (
            <div className="video-playback" style={{ marginBottom: '0px' }}>
              <video
                width="640"
                height="480"
                controls
                src={videoUrl}
                style={{ border: '1px solid #ccc' }}
              />
            </div>
          ) : (
            <div style={{ width: '640px', height: '360px', display: 'flex', alignItems: 'center', justifyContent: 'center', backgroundColor: '#f0f0f0', border: '1px solid #ccc' }}>
              No video available for this trip
            </div>
          )}

          {/* Timeline */}
          {tripEmotions.length > 0 ? (
            <div style={{ width: '640px' }}>
              {/* <h4>Emotion Timeline</h4> */}
              <div className="emotion-timeline" style={{ position: 'relative', height: '40px', backgroundColor: '#f0f0f0', borderRadius: '4px', marginTop: '5px', paddingBottom: '15px' }}>
                {tripEmotions.map((emotion, index) => {
                  const tripStart = new Date(selectedTrip.start_time).getTime();
                  const tripEnd = selectedTrip.end_time ? new Date(selectedTrip.end_time).getTime() : Date.now();
                  const emotionTime = new Date(emotion.timestamp).getTime();
                  const position = ((emotionTime - tripStart) / (tripEnd - tripStart)) * 100;
                  const colorArray = emotion.color.split(',').map(Number);

                  return (
                    <div
                      key={index}
                      title={`${toZonedTime(new Date(emotion.timestamp), 'Etc/GMT').toLocaleTimeString('en-GB', { hour12: false })} - ${emotion.emotion}`}
                      style={{
                        position: 'absolute',
                        left: `${position-3}%`,
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
  );
};

export default TripPlayback;