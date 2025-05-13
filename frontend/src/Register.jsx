import React, { useState } from 'react';
import { useNavigate, Link } from 'react-router-dom';

function Register() {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const navigate = useNavigate();

  const handleSubmit = async e => {
    e.preventDefault();
    const res = await fetch('http://localhost:5000/api/register', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ email, password })
    });

    if (res.ok) {
      alert('Đăng ký thành công! Mời bạn đăng nhập.');
      navigate('/login');
    } else {
      const data = await res.json();
      alert(data.error || 'Đăng ký thất bại');
    }
  };

  const formStyle = {
    maxWidth: '400px',
    margin: '90px auto',
    padding: '30px',
    border: '1px solid #ccc',
    borderRadius: '8px',
    boxShadow: '0 0 10px rgba(0,0,0,0.1)',
    backgroundColor: '#fff'
  };

  const inputStyle = {
    width: '100%',
    padding: '12px',
    margin: '5px 0',
    fontSize: '16px',
    border: '1px solid #ccc',
    borderRadius: '4px'
  };

  const buttonStyle = {
    width: '100%',
    padding: '12px',
    marginTop: '10px',
    backgroundColor: '#61dafb',
    border: 'none',
    borderRadius: '4px',
    fontSize: '16px',
    cursor: 'pointer'
  };

  const linkStyle = {
    marginTop: '15px',
    textAlign: 'center',
    display: 'block'
  };

  return (
    <form onSubmit={handleSubmit} style={formStyle}>
      <h2 style={{ textAlign: 'center' }}>Register</h2>
      <input
        type="email"
        placeholder="Email"
        value={email}
        onChange={e => setEmail(e.target.value)}
        required
        style={inputStyle}
      />
      <input
        type="password"
        placeholder="Password"
        value={password}
        onChange={e => setPassword(e.target.value)}
        required
        style={inputStyle}
      />
      <button type="submit" style={buttonStyle}>Register</button>
      <p style={linkStyle}>
        Already have account? <Link to="/login">Login</Link>
      </p>
    </form>
  );
}

export default Register;
