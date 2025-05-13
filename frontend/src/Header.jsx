// src/Header.js
import React from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { isAuthenticated, logout } from './auth';

function Header() {
  const navigate = useNavigate();

  const handleLogout = () => {
    logout();
    navigate('/login');
  };

  const headerStyle = {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    padding: '15px 40px',
    backgroundColor: '#f5f5f5',
    color: 'white'
  };

  const titleStyle = {
    fontSize: '32px',
    fontWeight: 'bold',
    margin: 0
  };

  const buttonStyle = {
    padding: '8px 16px',
    fontSize: '16px',
    backgroundColor: '#4a90e2',
    border: 'none',
    borderRadius: '4px',
    cursor: 'pointer'
  };

  return (
    <header style={headerStyle}>
      <h1 style={titleStyle}>Det4Safe</h1>
      <div>
        {isAuthenticated() ? (
          <button style={buttonStyle} onClick={handleLogout}>Logout</button>
        ) : (
          <Link to="/login">
            <button style={buttonStyle}>Login</button>
          </Link>
        )}
      </div>
    </header>
  );
}

export default Header;
