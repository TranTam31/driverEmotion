// import React from 'react';
// import './App.css';
// import Dashboard from './Dashboard';

// function App() {
//   return (
//     <div className="App">
//       <header className="App-header">
//         <h1>Det4Safe</h1>
//       </header>
//       <main>
//         <Dashboard />
//       </main>
//     </div>
//   );
// }

// export default App;

// src/App.js
import React from 'react';
import './App.css';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Header from './Header';
import Dashboard from './Dashboard';
import Login from './Login';
import Register from './Register';
import ProtectedRoute from './ProtectedRoute';

function App() {
  return (
    <Router>
      <div className="App">
        <Header />
        <main>
          <Routes>
            <Route path="/login" element={<Login />} />
            <Route path="/register" element={<Register />} />
            <Route path="/" element={
              <ProtectedRoute>
                <Dashboard />
              </ProtectedRoute>
            } />
          </Routes>
        </main>
      </div>
    </Router>
  );
}

export default App;