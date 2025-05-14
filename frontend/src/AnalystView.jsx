// AnalystView.jsx
import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { PieChart, Pie, Cell, Legend } from 'recharts';

const COLORS = ['#00ac00', '#FFBB28', '#FF8042', '#00C49F', '#aa46be', '#ff6666', '#66ccff'];

const AnalystView = ({ driverId }) => {
  const [minDate, setMinDate] = useState('');
  const [maxDate, setMaxDate] = useState('');
  const [startDate, setStartDate] = useState('');
  const [endDate, setEndDate] = useState('');
  const [result, setResult] = useState(null);

  useEffect(() => {
    if (!driverId) return;
    axios.get(`http://localhost:5000/api/emotion-date-range/${driverId}`)
      .then(res => {
        const { min_date, max_date } = res.data;
        setMinDate(min_date);
        setMaxDate(max_date);
        setStartDate(min_date);
        setEndDate(max_date);
      })
      .catch(err => {
        console.error('Error fetching date range:', err);
      });
  }, [driverId]);

  const handleAnalyze = () => {
    if (!startDate || !endDate) return alert("Please select a date range.");
    axios.post('http://localhost:5000/api/emotion-stats-by-range', {
      driver_id: driverId,
      start_date: startDate,
      end_date: endDate
    }).then(res => {
      setResult(res.data);
    }).catch(err => {
      console.error('Error fetching stats:', err);
    });
  };

  return (
    <div>
      <h3>Emotion Analytics</h3>
      {minDate && maxDate ? (
        <div style={{
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            marginBottom: '20px',
            fontSize: '16px',
            fontWeight: '500'
        }}>
            <div style={{ marginBottom: '10px' }}>
            <label style={{ fontWeight: 'bold', marginRight: '5px' }}>From:</label>
            <input
                type="date"
                value={startDate}
                onChange={e => setStartDate(e.target.value)}
                min={minDate}
                max={endDate}
                style={{ padding: '6px', fontSize: '15px' }}
            />
            <label style={{ fontWeight: 'bold', marginLeft: '15px', marginRight: '5px' }}>To:</label>
            <input
                type="date"
                value={endDate}
                onChange={e => setEndDate(e.target.value)}
                min={startDate}
                max={maxDate}
                style={{ padding: '6px', fontSize: '15px' }}
            />
            <button
                onClick={handleAnalyze}
                style={{
                marginLeft: '15px',
                width: '100px',
                backgroundColor: '#f48a8a',
                color: '#fff',
                fontWeight: 'bold',
                border: 'none',
                borderRadius: '4px',
                cursor: 'pointer',
                fontSize: '15px'
                }}
            >
                Analyze
            </button>
            </div>
            <div style={{ fontSize: '15px', fontWeight: 'bold' }}>
            {/* Range: {format(toZonedTime(new Date(startDate), 'Etc/GMT'), 'dd/MM/yyyy')} — {format(toZonedTime(new Date(endDate), 'Etc/GMT'), 'dd/MM/yyyy')} */}
            </div>
        </div>
        ) : <p style={{ textAlign: 'center' }}>Loading available date range...</p>}

      {result && (
        <>
            {/* <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                <thead>
                    <tr>
                    <th style={{ border: '1px solid #ccc', padding: '8px' }}>Emotion</th>
                    <th style={{ border: '1px solid #ccc', padding: '8px' }}>%</th>
                    </tr>
                </thead>
                <tbody>
                    {result.map(e => (
                    <tr key={e.emotion} style={{ textAlign: 'center' }}>
                        <td style={{ border: '1px solid #ddd', padding: '8px' }}>{e.emotion}</td>
                        <td style={{ border: '1px solid #ddd', padding: '8px' }}>{e.percentage.toFixed(2)}%</td>
                    </tr>
                    ))}
                </tbody>
            </table> */}

            <div style={{ marginTop: '30px', display: 'flex', justifyContent: 'center' }}>
                <PieChart width={600} height={300}>
                <Pie
                    data={result}
                    dataKey="percentage"
                    nameKey="emotion"
                    cx="50%"
                    cy="50%"
                    outerRadius={100}
                    fill="#8884d8"
                    label={({ name, percent }) => `${name}: ${(percent * 100).toFixed(2)}%`}
                >
                    {result.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                    ))}
                </Pie>
                <Legend />
                </PieChart>
            </div>
          </>
      )}
    </div>
  );
};

export default AnalystView;
