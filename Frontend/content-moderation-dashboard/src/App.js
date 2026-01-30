// App.jsx - Updated with Login and Protected Routes

import React from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import LoginPage from './components/LoginPage';
import ContentModerationDashboard from './components/ContentModerationDashboard';
import ProtectedRoute from './components/ProtectedRoute';
import { isAuthenticated } from './utils/mockAuth';
import "./App.css";

function App() {
  return (
    <Router>
      <Routes>
        {/* Login Route */}
        <Route path="/login" element={<LoginPage />} />


        {/* Protected Dashboard Route */}
        <Route 
          path="/dashboard" 
          element={
            <ProtectedRoute>
              <ContentModerationDashboard />
            </ProtectedRoute>
          } 
        />

        {/* Default Route */}
        <Route 
          path="/" 
          element={
            <Navigate to={isAuthenticated() ? "/dashboard" : "/login"} replace />
          } 
        />

        {/* Catch All - Redirect to Login */}
        <Route path="*" element={<Navigate to="/login" replace />} />
      </Routes>
    </Router>
  );
}

export default App;