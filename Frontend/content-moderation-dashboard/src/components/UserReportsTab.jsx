// UserReportsTab.jsx - Updated to use getReportsWithVideos API

import React, { useState, useEffect } from 'react';
import { Flag, RefreshCw, AlertTriangle, CheckCircle, ArrowLeft, AlertCircle } from 'lucide-react';
import ReportCard from './ReportCard';
import ReportFilters from './ReportFilters';
import VideoPlayer from './VideoPlayer';
import VideoCarousel from './VideoCarousel';
import SecureLinkHandler from '../utils/secureLinkHandler';
import { extractLinkedPosts } from '../utils/reportsApi';

import { 
  fetchReportsWithVideos, 
  fetchMiniReportsWithVideos,
  getContentTypeFromPost,
  parseInteractiveVideos,
  parseMiniInteractiveVideos,
  getStatusString,
  getStatusConfig,
  updateReportStatus 
} from '../utils/reportsApi';

const UserReportsTab = () => {
  const [reports, setReports] = useState([]);
  const [filteredReports, setFilteredReports] = useState([]);
  const [groupedReports, setGroupedReports] = useState([]);
  const [selectedReport, setSelectedReport] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [selectedPostGroup, setSelectedPostGroup] = useState(null); 
  const [filters, setFilters] = useState({
    reportType: 'all',
    status: 'all',
    search: '',
    sortBy: 'newest'
  });

  // Fetch reports on mount
  useEffect(() => {
    loadReports();
  }, []);

  // Apply filters whenever reports or filters change
  useEffect(() => {
    applyFilters();
  }, [reports, filters]);

  const loadReports = async () => {
    setIsLoading(true);
    try {
      console.log('🔐 Fetching all reports...');
      
      // Fetch both report types in parallel
      const [snipResponse, miniResponse] = await Promise.all([
        fetchReportsWithVideos(),
        fetchMiniReportsWithVideos()
      ]);
      
      console.log('✅ SNIP/SHOT/SSUP Response:', snipResponse);
      console.log('✅ MINI Response:', miniResponse);

      let allReports = [];

      // Process SNIP/SHOT/SSUP reports
      if (snipResponse.isSuccess && snipResponse.data) {
        const reportsData = snipResponse.data.formattedReports || [];
        
        const mappedReports = reportsData.map(report => {
          const postDetails = report.postDetails || {};
          const contentType = getContentTypeFromPost(postDetails);
          const videos = parseInteractiveVideos(
            postDetails.interactivevideo, 
            contentType,
            postDetails.multipleposts
          );
        
          let linkedPosts = [];
          if (contentType === 'Interactive SNIP') {
            linkedPosts = extractLinkedPosts(videos);
          }
          
          return {
            id: report.id,
            postid: report.postid,
            contentType: contentType,
            reportType: report.reportType,
            comment: report.comment,
            email: report.email,
            phone: report.phone,
            userid: report.userid,
            createdAt: report.createdAt,
            updatedAt: report.updatedAt,
            status: getStatusString(report.status), // ✅ Convert numeric to string
            postDetails: {
              title: postDetails.title,
              coverfilename: postDetails.coverfilename,
              videofilename: postDetails.videofilename,
              post_type: postDetails.post_type,
              isinteractive: postDetails.isinteractive,
              isforinteractivevideo: postDetails.isforinteractivevideo,
              isforinteractiveimage: postDetails.isforinteractiveimage,
              multipleposts: postDetails.multipleposts,
              ispost: postDetails.ispost,
              interactivevideo: postDetails.interactivevideo
            },
            videos: videos,
            hasMultipleMedia: videos.length > 1,
            linkedPosts: linkedPosts, 
            sourceType: 'post'
          };
        });

        allReports = [...allReports, ...mappedReports];
      }

      // Process MINI reports
      if (miniResponse.isSuccess && miniResponse.data) {
        const miniReportsData = miniResponse.data.formattedReports || [];
        
        const mappedMiniReports = miniReportsData.map(report => {
          const flixDetails = report.flixDetails || {};
          const videos = parseMiniInteractiveVideos(flixDetails.interactivevideo);

          let linkedPosts = [];
          try {
            linkedPosts = extractLinkedPosts(flixDetails.interactiveVideo || []);
          } catch (e) {
            console.error("Failed to extract MINI linked posts", e);
          }

          
          return {
            id: `mini-${report.id}`,
            postid: report.postid,
            contentType: 'MINI',
            reportType: report.reportType,
            comment: report.comment,
            email: report.email,
            phone: report.phone,
            userid: report.userid,
            createdAt: report.createdAt,
            updatedAt: report.updatedAt,
            status: getStatusString(report.status), // ✅ Convert numeric to string
            postDetails: {
              title: flixDetails.title,
              coverfilename: flixDetails.coverfilename,
              videofilename: flixDetails.videofilename,
              duration: flixDetails.duration,
              description: flixDetails.description,
              interactivevideo: flixDetails.interactivevideo
            },
            videos: videos,
            hasMultipleMedia: videos.length > 1,
            linkedPosts: linkedPosts,
            sourceType: 'mini'
          };
        });

        allReports = [...allReports, ...mappedMiniReports];
      }

      // Sort all reports by date
      allReports.sort((a, b) => new Date(b.createdAt) - new Date(a.createdAt));

      setReports(allReports);
      console.log('📊 Loaded total reports:', allReports.length);
      console.log('📊 MINI reports:', allReports.filter(r => r.sourceType === 'mini').length);
      
    } catch (error) {
      console.error('❌ Failed to load reports:', error);
      alert(`Failed to load reports: ${error.message}`);
      setReports([]);
    } finally {
      setIsLoading(false);
    }
  };

  const applyFilters = () => {
    let filtered = [...reports];

    // Filter by report type
    if (filters.reportType !== 'all') {
      filtered = filtered.filter(r => r.reportType === filters.reportType);
    }

    // Filter by status
    if (filters.status !== 'all') {
      filtered = filtered.filter(r => r.status === filters.status);
    }

    // Filter by search
    if (filters.search) {
      const searchLower = filters.search.toLowerCase();
      filtered = filtered.filter(r =>
        r.postid.toString().includes(searchLower) ||
        (r.email || '').toLowerCase().includes(searchLower) ||
        (r.phone || '').includes(searchLower) ||
        (r.comment || '').toLowerCase().includes(searchLower) ||
        (r.postDetails?.title || '').toLowerCase().includes(searchLower)
      );
    }

    // Sort
    if (filters.sortBy === 'newest') {
      filtered.sort((a, b) => new Date(b.createdAt) - new Date(a.createdAt));
    } else {
      filtered.sort((a, b) => new Date(a.createdAt) - new Date(b.createdAt));
    }

    setFilteredReports(filtered);
    setGroupedReports(groupReportsByPostId(filtered)); // Add this
  };

  const handleViewDetails = (postGroup) => {
    console.log('👁️ Viewing post group:', postGroup);
    setSelectedPostGroup(postGroup);
    setSelectedReport(postGroup.latestReport); // Set to latest report initially
  };

  const handleReportResolved = async (reportId, action, sourceType) => {
    try {
      console.log('🔄 Updating report:', { reportId, action, sourceType });
      
      // Show loading state (optional)
      setIsLoading(true);
  
      // Call API to update status
      const response = await updateReportStatus(reportId, action, sourceType);
      
      if (response.success) {
        console.log('✅ Report updated successfully:', response);
        
        // Update local state to reflect changes
        setReports(prev => prev.map(r => {
          if (r.id === reportId) {
            return {
              ...r,
              status: getStatusString(response.data.status)
            };
          }
          return r;
        }));
        
        // Show success message
        alert(`Report ${action} successfully!`);
        
        // Close detail view and refresh
        setSelectedReport(null);
        setSelectedPostGroup(null);
        
        // Optionally reload all reports to get fresh data
        await loadReports();
        
      } else {
        console.error('❌ Failed to update report:', response.error);
        alert(`Failed to update report: ${response.error}`);
      }
      
    } catch (error) {
      console.error('❌ Error updating report:', error);
      alert(`Error: ${error.message}`);
    } finally {
      setIsLoading(false);
    }
  };

  const stats = {
    total: reports.length,
    pending: reports.filter(r => r.status === 'pending').length,
    reviewed: reports.filter(r => r.status === 'reviewed').length,
    resolved: reports.filter(r => r.status === 'resolved').length,
    rejected: reports.filter(r => r.status === 'rejected').length
  };

  // Add this helper function in UserReportsTab.jsx before the component
  const groupReportsByPostId = (reports) => {
    const grouped = reports.reduce((acc, report) => {
      // Group by both postid AND contentType (sourceType)
      const key = `${report.postid}-${report.sourceType || 'post'}`;
      if (!acc[key]) {
        acc[key] = [];
      }
      acc[key].push(report);
      return acc;
    }, {});
  
    // Convert to array and sort by most recent report
    return Object.entries(grouped).map(([key, reports]) => ({
      postid: reports[0].postid,
      sourceType: reports[0].sourceType,
      contentType: reports[0].contentType,
      key: key, // Unique key for React
      reports: reports.sort((a, b) => new Date(b.createdAt) - new Date(a.createdAt)),
      latestReport: reports[0],
      reportCount: reports.length
    })).sort((a, b) => new Date(b.latestReport.createdAt) - new Date(a.latestReport.createdAt));
  };


  return (
    <div className="flex-1 flex flex-col h-full overflow-hidden">
      <div className="flex-1 flex overflow-hidden">
        {/* Reports List Sidebar */}
        <div className="w-80 bg-white border-r border-gray-200 flex flex-col max-h-screen">
          
          {/* Sticky header area: stats + filters */}
          <div className="sticky top-0 bg-white z-10">
            <div className="p-4 border-b border-gray-200">
              <div className="flex items-center justify-between mb-4">
                <h2 className="text-sm font-semibold text-gray-900">Reports Overview</h2>
                <button
                  onClick={loadReports}
                  disabled={isLoading}
                  className="p-1.5 text-gray-600 hover:text-gray-900 hover:bg-gray-100 rounded-md transition-colors disabled:opacity-50"
                  title="Refresh"
                >
                  <RefreshCw className={`w-4 h-4 ${isLoading ? 'animate-spin' : ''}`} />
                </button>
              </div>

              <div className="grid grid-cols-3 gap-2">
  <div className="p-3 bg-gray-50 rounded-lg border border-gray-200">
    <div className="text-xl font-semibold text-gray-900">{stats.total}</div>
    <div className="text-xs text-gray-600">Total</div>
  </div>
  <div className="p-3 bg-orange-50 rounded-lg border border-orange-200">
    <div className="text-xl font-semibold text-orange-600">{stats.pending}</div>
    <div className="text-xs text-orange-600">Pending</div>
  </div>
  <div className="p-3 bg-green-50 rounded-lg border border-green-200">
    <div className="text-xl font-semibold text-green-600">{stats.resolved}</div>
    <div className="text-xs text-green-600">Resolved</div>
  </div>
</div>
            </div>

            <div className="p-4 border-b border-gray-200">
              <ReportFilters filters={filters} onFilterChange={setFilters} />
            </div>
          </div>

          {/* List (scrollable) */}
          <div className="flex-1 overflow-y-auto">
  {isLoading ? (
    <div className="p-8 text-center">
      <RefreshCw className="w-8 h-8 mx-auto mb-3 animate-spin text-gray-400" />
      <p className="text-sm text-gray-500">Loading reports...</p>
    </div>
  ) : groupedReports.length === 0 ? (
    <div className="p-8 text-center text-gray-500">
      <Flag className="w-12 h-12 mx-auto mb-3 text-gray-300" />
      <p className="text-sm">No reports found</p>
      {(filters.reportType !== 'all' || filters.status !== 'all' || filters.search) && (
        <p className="text-xs mt-2">Try adjusting your filters</p>
      )}
    </div>
  ) : (
    groupedReports.map(postGroup => (
      <ReportCard
        key={postGroup.key}
        postGroup={postGroup}
        onViewDetails={handleViewDetails}
        isSelected={selectedPostGroup?.key === postGroup.key}
      />
    ))
  )}
</div>
        </div>

        {/* Preview Area */}
        <div className="flex-1 flex items-center justify-center bg-gray-50">
          {selectedReport ? (
            <div className="w-full h-full overflow-y-auto p-6">
              <div className="max-w-3xl mx-auto space-y-6">
                {/* Back Button */}
                <button 
                  onClick={() => setSelectedReport(null)}
                  className="inline-flex items-center gap-2 text-sm text-gray-600 hover:text-gray-900 mb-4"
                >
                  <ArrowLeft className="w-4 h-4" />
                  Back to reports
                </button>

                {/* Content Preview - Handle Multiple Media and Different Types */}
                <div className="space-y-4">
  {selectedReport.hasMultipleMedia && (
    <div className="bg-blue-50 border border-blue-200 rounded-lg p-3">
      <p className="text-sm text-blue-700 font-medium">
        {selectedReport.contentType === 'Interactive SNIP' ? '🎬 Interactive SNIP' : `📹 ${selectedReport.contentType}`} • {selectedReport.videos.length} Media Files
      </p>
    </div>
  )}
  
  {selectedReport.videos && selectedReport.videos.length > 0 ? (
  <VideoCarousel 
    key={selectedReport.id} // Add this line
    videos={selectedReport.videos} 
    contentType={selectedReport.contentType}
  />
  ) : selectedReport.postDetails?.coverfilename ? (
    <div className="rounded-lg overflow-hidden bg-black flex items-center justify-center">
      <img 
        src={selectedReport.postDetails.coverfilename} 
        className="max-h-[400px] w-auto object-contain" 
        alt="Post thumbnail"
      />
    </div>
  ) : (
    <div className="w-full h-64 flex items-center justify-center bg-gray-200 text-gray-500 rounded-lg">
      <div className="text-center">
        <Flag className="w-12 h-12 mx-auto mb-2 text-gray-400" />
        <p className="text-sm">No Media Available</p>
      </div>
    </div>
  )}
</div>


{/* Multiple Reports Dropdown - Simple One Line */}
{selectedPostGroup && selectedPostGroup.reportCount > 1 && (
  <div className="bg-amber-50 border border-amber-200 rounded-lg p-4">
    <div className="flex items-center gap-3 mb-3">
      <AlertCircle className="w-5 h-5 text-amber-600 flex-shrink-0" />
      <div className="flex items-center gap-3 flex-1">
        <span className="text-sm font-semibold text-amber-900 whitespace-nowrap">
          Multiple Reports ({selectedPostGroup.reportCount}):
        </span>
        <select
          value={selectedReport?.id || ''}
          onChange={(e) => {
            const reportId = e.target.value;
            const selected = selectedPostGroup.reports.find(r => r.id.toString() === reportId.toString());
            if (selected) {
              console.log('🔄 Selected report changed:', selected);
              setSelectedReport(selected);
            }
          }}
          className="flex-1 px-3 py-1.5 bg-white border border-amber-300 rounded text-sm text-gray-900 focus:outline-none focus:ring-2 focus:ring-amber-500 focus:border-transparent cursor-pointer"
        >
          {selectedPostGroup.reports.map((report, index) => {
            const date = new Date(report.createdAt);
            const dateStr = date.toLocaleDateString('en-US', { 
              month: 'short', 
              day: 'numeric',
              hour: '2-digit',
              minute: '2-digit'
            });
            
            return (
              <option key={report.id} value={report.id}>
                Report #{index + 1} - {report.reportType} - {dateStr}
              </option>
            );
          })}
        </select>
      </div>
    </div>
  </div>
)}

                {/* Report Details Card */}
<div className="bg-white rounded-lg border border-gray-200 p-6 space-y-4">
  {/* Header Info */}
  <div className="border-b border-gray-200 pb-4">
    <div className="flex items-start justify-between mb-2">
      <div>
        <h2 className="text-lg font-semibold text-gray-900">Report #{selectedReport.id}</h2>
        <p className="text-sm text-gray-500">Post ID: {selectedReport.postid}</p>
        {selectedReport.postDetails?.title && (
          <p className="text-sm text-gray-600 mt-1">{selectedReport.postDetails.title}</p>
        )}
      </div>
      <span className="px-3 py-1 bg-red-100 text-red-700 text-sm font-medium rounded-full">
        {selectedReport.reportType}
      </span>
    </div>
    <p className="text-xs text-gray-400">
      {new Date(selectedReport.createdAt).toLocaleString()}
    </p>
  </div>

  {/* Post Metadata */}
  <div>
    <h3 className="text-sm font-medium text-gray-700 mb-2">Post Details</h3>
    <div className="bg-gray-50 p-3 rounded-lg space-y-1 text-sm">
      <div className="flex justify-between">
        <span className="text-gray-600">Type:</span>
        <span className="text-gray-900 font-medium">{selectedReport.contentType}</span>
      </div>
      <div className="flex justify-between">
        <span className="text-gray-600">Interactive:</span>
        <span className="text-gray-900 font-medium">
          {selectedReport.postDetails?.isinteractive ? 'Yes' : 'No'}
        </span>
      </div>
      {selectedReport.hasMultipleMedia && (
        <div className="flex justify-between">
          <span className="text-gray-600">Media Files:</span>
          <span className="text-gray-900 font-medium">
            {selectedReport.videos.length}
          </span>
        </div>
      )}
    </div>
  </div>

  {/* Linked Posts Section */}
  {selectedReport.linkedPosts && selectedReport.linkedPosts.length > 0 && (
    <div>
      <h3 className="text-sm font-medium text-gray-700 mb-2">
        Linked Posts ({selectedReport.linkedPosts.length})
      </h3>
      <div className="bg-blue-50 border border-blue-200 rounded-lg p-3 space-y-2">
        <p className="text-xs text-blue-600 mb-2">
          This Interactive SNIP contains buttons linking to other posts
        </p>
        {selectedReport.linkedPosts.map((linked, index) => {
          const webUrl = SecureLinkHandler.generateSnipLink(selectedReport.postid);
          return (
            <div key={index} className="flex items-center justify-between bg-white p-2 rounded border border-blue-200">
              <div className="flex-1">
                <p className="text-sm font-medium text-gray-900">Post ID: {linked.id}</p>
                <p className="text-xs text-gray-500">Source: {selectedReport.postid}</p>
              </div>
              <a
  href={webUrl}
  target="_blank"
  rel="noopener noreferrer"
  className="px-3 py-1.5 bg-blue-600 text-white text-xs font-medium rounded hover:bg-blue-700 transition-colors flex items-center gap-1"
>
  <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} 
      d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14" 
    />
  </svg>
  View on Web
</a>
            </div>
          );
        })}
      </div>
    </div>
  )}

  {/* Report Comment */}
  {selectedReport.comment && (
    <div>
      <h3 className="text-sm font-medium text-gray-700 mb-1">Reporter's Comment</h3>
      <p className="text-sm text-gray-600 bg-gray-50 p-3 rounded-lg">
        {selectedReport.comment}
      </p>
    </div>
  )}

  {/* Reporter Details */}
  <div>
    <h3 className="text-sm font-medium text-gray-700 mb-2">Reporter Information</h3>
    <div className="bg-gray-50 p-3 rounded-lg space-y-1 text-sm">
      <div className="flex justify-between">
        <span className="text-gray-600">Email:</span>
        <span className="text-gray-900 font-medium">
          {selectedReport.email || 'Not provided'}
        </span>
      </div>
      <div className="flex justify-between">
        <span className="text-gray-600">Phone:</span>
        <span className="text-gray-900 font-medium">
          {selectedReport.phone || 'Not provided'}
        </span>
      </div>
      <div className="flex justify-between">
        <span className="text-gray-600">User ID:</span>
        <span className="text-gray-900 font-medium">
          {selectedReport.userid}
        </span>
      </div>
    </div>
  </div>
</div>

                {/* Action Buttons */}
<div className="flex gap-3">
  <button
    onClick={() => handleReportResolved(selectedReport.id, 'rejected', selectedReport.sourceType)}
    disabled={isLoading}
    className="flex-1 px-4 py-2.5 bg-gray-600 text-white rounded-lg hover:bg-gray-700 transition-colors font-medium disabled:opacity-50 disabled:cursor-not-allowed"
  >
    Reject Report
  </button>

  <button
    onClick={() => handleReportResolved(selectedReport.id, 'reviewed', selectedReport.sourceType)}
    disabled={isLoading}
    className="flex-1 px-4 py-2.5 bg-yellow-600 text-white rounded-lg hover:bg-yellow-700 transition-colors font-medium disabled:opacity-50 disabled:cursor-not-allowed"
  >
    Mark as Reviewed
  </button>

  <button
    onClick={() => handleReportResolved(selectedReport.id, 'resolved', selectedReport.sourceType)}
    disabled={isLoading}
    className="flex-1 px-4 py-2.5 bg-red-600 text-white rounded-lg hover:bg-red-700 transition-colors font-medium disabled:opacity-50 disabled:cursor-not-allowed"
  >
    Remove Post
  </button>
</div>
              </div>
            </div>
          ) : (
            <div className="text-center p-8">
              <div className="w-16 h-16 bg-gray-100 rounded-lg flex items-center justify-center mx-auto mb-4">
                <Flag className="w-8 h-8 text-gray-400" />
              </div>
              <h3 className="text-base font-medium text-gray-700 mb-1">No Report Selected</h3>
              <p className="text-sm text-gray-500">Select a report from the list to review</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default UserReportsTab;