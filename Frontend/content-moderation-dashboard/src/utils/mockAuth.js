/**
 * Set authentication token
 */
export const setToken = (token) => {
    localStorage.setItem('token', token);
  };
  
  /**
   * Check if user is authenticated
   */
  export const isAuthenticated = () => {
    return localStorage.getItem("token") !== null;
  };
  
  /**
   * Get current user info
   */
  export const getCurrentUser = () => {
    if (!isAuthenticated()) {
      return null;
    }
    
    return {
      email: localStorage.getItem('userEmail'),
      role: localStorage.getItem('userRole'),
      fullName: localStorage.getItem('fullName')
    };
  };
  
  /**
   * Logout user
   */
  export const logout = () => {
    localStorage.removeItem('isAuthenticated');
    localStorage.removeItem('userEmail');
    localStorage.removeItem('userRole');
    localStorage.removeItem('rememberMe');
    localStorage.removeItem('token');
    localStorage.removeItem('fullName');
  };
  
  /**
   * Mock user credentials for testing
   */