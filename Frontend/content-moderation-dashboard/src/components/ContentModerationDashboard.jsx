import React, { useState, useEffect } from 'react';
import { Eye, CheckCircle, XCircle, AlertTriangle, FileText, ChevronDown, ChevronUp, RefreshCw, Flag } from 'lucide-react';
import axios from 'axios';
import { useNavigate } from 'react-router-dom';
import { logout, getCurrentUser } from '../utils/mockAuth';
import { LogOut, User, Shield } from 'lucide-react';
import UserReportsTab from './UserReportsTab';

const API_BASE_URL = 'http://localhost:8000';

const flagLabels = {
  nudity: 'Nudity/NSFW',
  violence: 'Violence',
  hateSpeech: 'Hate Speech',
  communityTargeted: 'Community Targeted/Racial Slurs',
  political: 'Political Agenda/Propaganda'
};

const ContentModerationDashboard = () => {
  const [activeTab, setActiveTab] = useState('ai-detection'); // 'ai-detection' or 'user-reports'
  const [selectedContent, setSelectedContent] = useState(null);
  const [contentList, setContentList] = useState([]);
  const [filterType, setFilterType] = useState('all');
  const [isCaptionExpanded, setIsCaptionExpanded] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [lastFetch, setLastFetch] = useState(null);

  const navigate = useNavigate();
  const currentUser = getCurrentUser();

  // Fetch moderation results from backend
  const fetchContent = async () => {
    setIsLoading(true);
    try {
      const response = await axios.get(`${API_BASE_URL}/api/content`);
      
      if (response.data.success) {
        setContentList(response.data.content);
        setLastFetch(new Date());
        
        // Auto-select first item if none selected
        if (!selectedContent && response.data.content.length > 0) {
          setSelectedContent(response.data.content[0]);
        }
      }
    } catch (error) {
      console.error('Failed to fetch content:', error);
      alert('Failed to load moderation results. Make sure backend is running.');
    } finally {
      setIsLoading(false);
    }
  };

  // Fetch on component mount (only for AI detection tab)
  useEffect(() => {
    if (activeTab === 'ai-detection') {
      fetchContent();
      
      // Auto-refresh every 30 seconds
      const interval = setInterval(fetchContent, 30000);
      return () => clearInterval(interval);
    }
  }, [activeTab]);

  const getContentTypeIcon = (type) => {
    // Using img tags to load SVG icons from public/assets
    const iconClass = "w-4 h-4";
    
    switch (type) {
      case 'SNIP': 
        return <img src="/assets/snip-icon.svg" alt="Snip" className={iconClass} />;
      case 'SHOT': 
        return <img src="/assets/shot-icon.svg" alt="Shot" className={iconClass} />;
      case 'MINI': 
        return <img src="/assets/mini-icon.svg" alt="Mini" className={iconClass} />;
      case 'SSUP': 
        return <img src="/assets/ssup-icon.svg" alt="Ssup" className={iconClass} />;
      default: 
        return <Eye className="w-4 h-4" />;
    }
  };

  const handleLogout = () => {
    logout();
    navigate('/login');
  };

  const getContentTypeColor = (type) => {
    switch (type) {
      case 'SNIP': return 'from-purple-500 to-pink-500';
      case 'SHOT': return 'from-blue-500 to-cyan-500';
      case 'MINI': return 'from-indigo-500 to-purple-500';
      case 'SSUP': return 'from-pink-500 to-rose-500';
      default: return 'from-gray-500 to-gray-600';
    }
  };

  const handleReview = async (contentId, decision) => {
    try {
      // Remove from backend
      await axios.delete(`${API_BASE_URL}/api/content/${contentId}`);
      
      // Remove from local state
      setContentList(prev => prev.filter(item => item.id !== contentId));
      
      const remainingContent = contentList.filter(item => item.id !== contentId);
      if (remainingContent.length > 0) {
        setSelectedContent(remainingContent[0]);
      } else {
        setSelectedContent(null);
      }
      
      console.log(`Content ${contentId} ${decision}`);
    } catch (error) {
      console.error('Failed to remove content:', error);
    }
  };

  const filteredContent = contentList.filter(item => {
    if (filterType !== 'all' && item.type !== filterType) return false;
    return true;
  });

  const formatTimestamp = (timestamp) => {
    return new Date(timestamp).toLocaleString();
  };

  const getRiskLevel = (flags) => {
    const detections = Object.values(flags).filter(f => f.detected).length;
  
    if (detections >= 4) {
      return { 
        level: 'VERY HIGH', 
        color: 'text-red-600', 
        bg: 'bg-red-50', 
        border: 'border-red-200' 
      };
    }
  
    if (detections === 3) {
      return { 
        level: 'HIGH', 
        color: 'text-red-600', 
        bg: 'bg-red-50', 
        border: 'border-red-200' 
      };
    }
  
    if (detections === 2) {
      return { 
        level: 'MEDIUM', 
        color: 'text-orange-600', 
        bg: 'bg-orange-50', 
        border: 'border-orange-200' 
      };
    }
  
    return { 
      level: 'LOW', 
      color: 'text-gray-600', 
      bg: 'bg-gray-50', 
      border: 'border-gray-200' 
    };
  };

  useEffect(() => {
    if (!selectedContent && filteredContent.length > 0) {
      setSelectedContent(filteredContent[0]);
    } else if (selectedContent && !filteredContent.find(item => item.id === selectedContent.id)) {
      setSelectedContent(filteredContent[0] || null);
    }
  }, [filteredContent]);

  useEffect(() => {
    setIsCaptionExpanded(false);
  }, [selectedContent]);

  const stats = {
    pending: contentList.length,
    highRisk: contentList.filter(item => getRiskLevel(item.flags).level === 'HIGH' || getRiskLevel(item.flags).level === 'VERY HIGH').length,
    reviewed: 0
  };

  // Render User Reports Tab
  if (activeTab === 'user-reports') {
    return (
      <div className="min-h-screen bg-gray-50 flex flex-col">
        {/* Header */}
        <div className="bg-white border-b border-gray-200 px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="w-12 h-12 bg-white rounded-lg overflow-hidden border border-gray-200">
                <img 
                  src="/assets/logo.png" 
                  alt="Bigshorts Logo" 
                  className="w-full h-full object-cover"
                />
              </div>
              <div>
                <h1 className="text-xl font-semibold text-gray-900">Bigshorts</h1>
                <p className="text-sm text-gray-500">Content Moderation Portal</p>
              </div>
            </div>

            {/* Tab Switcher - CENTER */}
            <div className="flex items-center gap-2 bg-gray-100 p-1 rounded-lg">
              <button
                onClick={() => setActiveTab('ai-detection')}
                className="px-4 py-2 text-sm font-medium text-gray-600 rounded-md transition-all"
              >
                AI Detection
              </button>
              <button
                onClick={() => setActiveTab('user-reports')}
                className="px-4 py-2 text-sm font-medium bg-white text-gray-900 rounded-md shadow-sm"
              >
                User Reports
              </button>
            </div>

            {/* User Info */}
            {currentUser && (
              <div className="flex items-center gap-3">
                <div className="flex items-center gap-2">
                  <div className="w-8 h-8 bg-gray-100 rounded-full flex items-center justify-center">
                    <User className="w-4 h-4 text-gray-600" />
                  </div>
                  <div>
                    <p className="text-sm font-medium text-gray-900">{currentUser.fullName}</p>
                    <p className="text-xs text-gray-500">{currentUser.role}</p>
                  </div>
                </div>
                <button
                  onClick={handleLogout}
                  className="p-2 text-gray-600 hover:text-gray-900 hover:bg-gray-100 rounded-lg transition-colors"
                  title="Logout"
                >
                  <LogOut className="w-5 h-5" />
                </button>
              </div>
            )}
          </div>
        </div>

        {/* User Reports Content */}
        <UserReportsTab />
      </div>
    );
  }

  // AI Detection Tab
  return (
    <div className="min-h-screen bg-gray-50 flex flex-col">
      {/* Header */}
      <div className="bg-white border-b border-gray-200 px-6 py-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-12 h-12 bg-white rounded-lg overflow-hidden border border-gray-200">
              <img 
                src="/assets/logo.png" 
                alt="Bigshorts Logo" 
                className="w-full h-full object-cover"
              />
            </div>
            <div>
              <h1 className="text-xl font-semibold text-gray-900">Bigshorts</h1>
              <p className="text-sm text-gray-500">Content Moderation Portal</p>
            </div>
          </div>

          {/* Tab Switcher - CENTER */}
          <div className="flex items-center gap-2 bg-gray-100 p-1 rounded-lg">
            <button
              onClick={() => setActiveTab('ai-detection')}
              className="px-4 py-2 text-sm font-medium bg-white text-gray-900 rounded-md shadow-sm"
            >
              AI Detection
            </button>
            
            <button
              onClick={() => setActiveTab('user-reports')}
              className="px-4 py-2 text-sm font-medium text-gray-600 rounded-md transition-all"
            >
              User Reports
            </button>
          </div>

          {/* User Info */}
          {currentUser && (
            <div className="flex items-center gap-3">
              <div className="flex items-center gap-2">
                <div className="w-8 h-8 bg-gray-100 rounded-full flex items-center justify-center">
                  <User className="w-4 h-4 text-gray-600" />
                </div>
                <div>
                  <p className="text-sm font-medium text-gray-900">{currentUser.fullName}</p>
                  <p className="text-xs text-gray-500">{currentUser.role}</p>
                </div>
              </div>
              <button
                onClick={handleLogout}
                className="p-2 text-gray-600 hover:text-gray-900 hover:bg-gray-100 rounded-lg transition-colors"
                title="Logout"
              >
                <LogOut className="w-5 h-5" />
              </button>
            </div>
          )}
        </div>
      </div>

      <div className="flex-1 flex flex-col overflow-hidden">
        <div className="flex-1 flex overflow-hidden">
          {/* Sidebar */}
          <div className="w-80 bg-white border-r border-gray-200 flex flex-col max-h-screen">
            {/* Stats */}
            <div className="sticky top-0 bg-white z-10">
            <div className="p-4 border-b border-gray-200">
              <div className="flex items-center justify-between mb-4">
                <h2 className="text-sm font-semibold text-gray-900">Queue Overview</h2>
                <button
                  onClick={fetchContent}
                  disabled={isLoading}
                  className="p-1.5 text-gray-600 hover:text-gray-900 hover:bg-gray-100 rounded-md transition-colors disabled:opacity-50"
                  title="Refresh"
                >
                  <RefreshCw className={`w-4 h-4 ${isLoading ? 'animate-spin' : ''}`} />
                </button>
              </div>
              
              <div className="grid grid-cols-3 gap-2">
                <div className="p-3 bg-gray-50 rounded-lg border border-gray-200">
                  <div className="text-xl font-semibold text-gray-900">{stats.pending}</div>
                  <div className="text-xs text-gray-600">Pending</div>
                </div>
                <div className="p-3 bg-red-50 rounded-lg border border-red-200">
                  <div className="text-xl font-semibold text-red-600">{stats.highRisk}</div>
                  <div className="text-xs text-red-600">High Risk</div>
                </div>
                <div className="p-3 bg-gray-50 rounded-lg border border-gray-200">
                  <div className="text-xl font-semibold text-gray-900">{stats.reviewed}</div>
                  <div className="text-xs text-gray-600">Reviewed</div>
                </div>
              </div>
            </div>

            {/* Filter */}
            <div className="p-4 border-b border-gray-200">
              <label className="text-xs font-medium text-gray-700 mb-2 block">Filter by Type</label>
              <select
                value={filterType}
                onChange={(e) => setFilterType(e.target.value)}
                className="w-full px-3 py-2 bg-white border border-gray-300 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-purple-500 focus:border-transparent"
              >
                <option value="all">All Content</option>
                <option value="SNIP">Snips</option>
                <option value="SHOT">Shots</option>
                <option value="MINI">Minis</option>
                <option value="SSUP">Ssups</option>
              </select>
            </div>
            </div>

            {/* Content List */}
            <div className="flex-1 overflow-y-auto">
              {isLoading && contentList.length === 0 ? (
                <div className="p-8 text-center">
                  <RefreshCw className="w-8 h-8 mx-auto mb-3 animate-spin text-gray-400" />
                  <p className="text-sm text-gray-500">Loading content...</p>
                </div>
              ) : filteredContent.length === 0 ? (
                <div className="p-8 text-center">
                  <Eye className="w-12 h-12 mx-auto mb-3 text-gray-300" />
                  <p className="text-sm text-gray-500">No content to review</p>
                </div>
              ) : (
                <div className="divide-y divide-gray-100">
                  {filteredContent.map((item) => {
                    const risk = getRiskLevel(item.flags);
                    const isSelected = selectedContent?.id === item.id;
                    
                    return (
                      <button
                        key={item.id}
                        onClick={() => setSelectedContent(item)}
                        className={`w-full p-4 text-left transition-colors ${
                          isSelected ? 'bg-purple-50' : 'hover:bg-gray-50'
                        }`}
                      >
                        <div className="flex items-start gap-3">
                          <div className={`w-8 h-8 bg-gradient-to-br ${getContentTypeColor(item.type)} rounded-lg flex items-center justify-center flex-shrink-0`}>
                            {getContentTypeIcon(item.type)}
                          </div>
                          
                          <div className="flex-1 min-w-0">
                            <div className="flex items-center gap-2 mb-1">
                              <span className="text-sm font-medium text-gray-900">{item.title}</span>
                              <span className={`px-2 py-0.5 text-xs font-medium rounded ${risk.bg} ${risk.color}`}>
                                {risk.level}
                              </span>
                            </div>
                            <p className="text-xs text-gray-500">{formatTimestamp(item.timestamp)}</p>
                          </div>
                        </div>
                      </button>
                    );
                  })}
                </div>
              )}
            </div>
          </div>

          {/* Main Content Area */}
          {selectedContent ? (
            <>
              <div className="flex-1 flex overflow-hidden">
                <div className="flex-1 p-8 flex flex-col items-center justify-center overflow-y-auto bg-gray-50">
                  {selectedContent.mediaUrl ? (
                    <div className="relative group mb-6 max-w-4xl w-full">
                      {selectedContent.type === 'SHOT' ? (
                        <div className="relative bg-black rounded-lg overflow-hidden w-[500px] h-[500px] mx-auto">
                          <img
                            src={`${API_BASE_URL}${selectedContent.mediaUrl}`}
                            alt={selectedContent.title}
                            className="w-full h-full object-contain"
                          />
                        </div>
                      ) : selectedContent.isVertical ? (
                        <div className="relative bg-black rounded-lg overflow-hidden w-[360px] h-[640px] mx-auto">
                          <video
                            key={selectedContent.id}
                            controls
                            className="w-full h-full object-contain"
                            src={`${API_BASE_URL}${selectedContent.mediaUrl}`}
                          >
                            Your browser does not support video playback.
                          </video>
                        </div>
                      ) : (
                        <div className="relative bg-black rounded-lg overflow-hidden w-full aspect-video">
                          <video
                            key={selectedContent.id}
                            controls
                            className="w-full h-full object-contain"
                            src={`${API_BASE_URL}${selectedContent.mediaUrl}`}
                          >
                            Your browser does not support video playback.
                          </video>
                        </div>
                      )}
                    </div>
                  ) : (
                    <div className="relative group mb-6">
                      <div className="relative bg-gray-900 rounded-lg w-[500px] h-[400px] flex items-center justify-center">
                        <div className="text-center text-gray-400">
                          {getContentTypeIcon(selectedContent.type)}
                          <p className="text-sm mt-4">Media Not Available</p>
                        </div>
                      </div>
                    </div>
                  )}
                  
                  {selectedContent.caption && (
                    <div className="w-full max-w-4xl mb-6">
                      <div className="bg-white rounded-lg border border-gray-200 overflow-hidden">
                        <button 
                          onClick={() => setIsCaptionExpanded(!isCaptionExpanded)}
                          className="w-full px-4 py-3 flex items-center gap-2 hover:bg-gray-50 transition-colors cursor-pointer border-b border-gray-200"
                        >
                          <FileText className="w-4 h-4 text-gray-600" />
                          <h3 className="font-medium text-gray-900">Caption / Text Content</h3>
                          {(selectedContent.flags.hateSpeech.detected || selectedContent.flags.communityTargeted.detected) && (
                            <span className="ml-2 px-2 py-0.5 bg-red-100 text-red-600 text-xs font-medium rounded">
                              Flagged
                            </span>
                          )}
                          <div className="ml-auto">
                            {isCaptionExpanded ? <ChevronUp className="w-4 h-4 text-gray-600" /> : <ChevronDown className="w-4 h-4 text-gray-600" />}
                          </div>
                        </button>
                        {isCaptionExpanded && (
                          <div className="p-4 bg-gray-50">
                            <p className="text-sm text-gray-700 leading-relaxed whitespace-pre-wrap">{selectedContent.caption}</p>
                          </div>
                        )}
                      </div>
                    </div>
                  )}
                  
                  <div className="flex justify-center gap-3 mt-4">
                    <button
                      onClick={() => handleReview(selectedContent.id, 'approved')}
                      className="flex items-center gap-2 px-6 py-2.5 bg-gray-900 text-white rounded-lg hover:bg-gray-800 transition-colors font-medium"
                    >
                      <CheckCircle className="w-5 h-5" />
                      Approve
                    </button>
                    <button
                      onClick={() => handleReview(selectedContent.id, 'rejected')}
                      className="flex items-center gap-2 px-6 py-2.5 bg-red-600 text-white rounded-lg hover:bg-red-700 transition-colors font-medium"
                    >
                      <XCircle className="w-5 h-5" />
                      Reject
                    </button>
                  </div>
                </div>

                {/* Detection Results Panel */}
                <div className="w-80 bg-white border-l border-gray-200 p-6 overflow-y-auto">
                  <div className="mb-6">
                    <h3 className="font-semibold text-gray-900 mb-1">AI Detection Results</h3>
                    <p className="text-xs text-gray-500">Automated content analysis</p>
                  </div>
                  
                  <div className="space-y-3 mb-6">
                    {Object.entries(selectedContent.flags).map(([key, flag]) => (
                      <div key={key} className={`p-3 border rounded-lg ${flag.detected ? 'bg-red-50 border-red-200' : 'bg-gray-50 border-gray-200'}`}>
                        <div className="flex items-center justify-between mb-2">
                          <span className="font-medium text-sm text-gray-900">{flagLabels[key]}</span>
                          <div className={`w-2 h-2 rounded-full ${flag.detected ? 'bg-red-500' : 'bg-gray-400'}`} />
                        </div>
                        
                        <div className="flex items-center justify-between text-xs mb-2">
                          <span className={`font-medium ${flag.detected ? 'text-red-600' : 'text-gray-600'}`}>
                            {flag.detected ? 'Detected' : 'Clean'}
                          </span>
                          <span className="text-gray-500">
                            {Math.round(flag.confidence * 100)}%
                          </span>
                        </div>
                        
                        <div className="relative bg-gray-200 rounded-full h-1.5 overflow-hidden">
                          <div 
                            className={`absolute top-0 left-0 h-full rounded-full transition-all ${flag.detected ? 'bg-red-500' : 'bg-gray-400'}`}
                            style={{ width: `${flag.confidence * 100}%` }}
                          />
                        </div>
                      </div>
                    ))}
                  </div>
                  
                  <div className="p-3 bg-amber-50 border border-amber-200 rounded-lg">
                    <div className="flex items-start gap-2">
                      <AlertTriangle className="w-4 h-4 text-amber-600 flex-shrink-0 mt-0.5" />
                      <div>
                        <span className="font-medium text-xs text-amber-900 block mb-1">Guidelines</span>
                        <p className="text-xs text-amber-800">
                          Consider context and cultural sensitivity when reviewing.
                        </p>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </>
          ) : (
            <div className="flex-1 flex items-center justify-center bg-gray-50">
              <div className="text-center">
                <div className="w-16 h-16 bg-gray-100 rounded-lg flex items-center justify-center mx-auto mb-4">
                  <Eye className="w-8 h-8 text-gray-400" />
                </div>
                <p className="text-base font-medium text-gray-700 mb-1">No Content Selected</p>
                <p className="text-sm text-gray-500">Select an item from the queue to review</p>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default ContentModerationDashboard;