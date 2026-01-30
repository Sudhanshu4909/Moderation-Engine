// ProtectedRoute.jsx - Protected Route Component

import React from 'react';
import { Navigate } from 'react-router-dom';
import { isAuthenticated } from '../utils/mockAuth';

const ProtectedRoute = ({ children }) => {
  if (!isAuthenticated()) {
    // Redirect to login if not authenticated
    return <Navigate to="/login" replace />;
  }

  return children;
};

export default ProtectedRoute;