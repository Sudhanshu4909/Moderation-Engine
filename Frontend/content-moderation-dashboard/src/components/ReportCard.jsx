import React from "react";
import { Clock, AlertCircle } from "lucide-react";
import { getReportTypeConfig } from "../utils/reportsApi";

const ReportCard = ({ postGroup, onViewDetails, isSelected }) => {
  const { latestReport, reportCount } = postGroup;
  const reportTypeConfig = getReportTypeConfig(latestReport.reportType);

  const formatDate = (dateString) => {
    const date = new Date(dateString);
    return date.toLocaleDateString() + " • " + date.toLocaleTimeString();
  };

  const getContentTypeIcon = (type) => {
    const iconClass = "w-4 h-4";
    switch (type) {
      case "SNIP":
        return <img src="/assets/snip-icon.svg" alt="Snip" className={iconClass} />;
      case "SHOT":
        return <img src="/assets/shot-icon.svg" alt="Shot" className={iconClass} />;
      case "MINI":
        return <img src="/assets/mini-icon.svg" alt="Mini" className={iconClass} />;
      case "SSUP":
        return <img src="/assets/ssup-icon.svg" alt="Ssup" className={iconClass} />;
      default:
        return <img src="/assets/snip-icon.svg" alt="Default" className={iconClass} />;
    }
  };

  const getContentTypeColor = (type) => {
    switch (type) {
      case "SNIP":
        return "from-purple-500 to-pink-500";
      case "SHOT":
        return "from-blue-500 to-cyan-500";
      case "MINI":
        return "from-indigo-500 to-purple-500";
      case "SSUP":
        return "from-pink-500 to-rose-500";
      default:
        return "from-gray-500 to-gray-600";
    }
  };

  const getStatusBadge = (status) => {
    const styles = {
      pending: "bg-orange-100 text-orange-700",
      reviewed: "bg-yellow-100 text-yellow-700",
      resolved: "bg-green-100 text-green-700",
      rejected: "bg-gray-100 text-gray-600",
    };
    
    return (
      <span className={`px-2 py-0.5 text-xs font-medium rounded-full ${styles[status] || styles.pending}`}>
        {status.charAt(0).toUpperCase() + status.slice(1)}
      </span>
    );
  };

  // Get unique report types for this post
  const uniqueReportTypes = [...new Set(postGroup.reports.map(r => r.reportType))];

  return (
    <button
      onClick={() => onViewDetails(postGroup)}
      className={`w-full p-4 text-left transition-all border-b border-gray-100 ${
        isSelected ? "bg-purple-50 border-l-4 border-l-purple-600 shadow-sm" : "hover:bg-gray-50"
      }`}
    >
      <div className="flex items-start gap-3">
        
        {/* Icon with badge */}
        <div className="relative">
          <div
            className={`w-8 h-8 bg-gradient-to-br ${getContentTypeColor(
              latestReport.contentType
            )} rounded-lg flex items-center justify-center`}
          >
            {getContentTypeIcon(latestReport.contentType)}
          </div>
          {reportCount > 1 && (
            <div className="absolute -top-1 -right-1 w-5 h-5 bg-red-500 text-white text-xs font-bold rounded-full flex items-center justify-center">
              {reportCount}
            </div>
          )}
        </div>

        {/* Content */}
        <div className="flex-1 min-w-0">
          
          {/* Title = Content Type */}
          <div className="flex items-center justify-between mb-1">
            <span className="text-sm font-semibold text-gray-900">
              {latestReport.contentType}
            </span>
            {getStatusBadge(latestReport.status || "pending")}
          </div>

          {/* Multiple reports indicator */}
          {reportCount > 1 ? (
            <div className="flex items-center gap-1 mb-1">
              <AlertCircle className="w-3 h-3 text-red-500" />
              <span className="text-xs font-semibold text-red-600">
                {reportCount} Reports
              </span>
            </div>
          ) : (
            <div className="text-xs font-medium text-gray-600 mb-1">
              {latestReport.reportType}
            </div>
          )}

          {/* Show all unique report types if multiple */}
          {reportCount > 1 && (
            <div className="text-xs text-gray-500 mb-1">
              {uniqueReportTypes.slice(0, 2).join(', ')}
              {uniqueReportTypes.length > 2 && ` +${uniqueReportTypes.length - 2} more`}
            </div>
          )}

          {/* Post ID */}
          <p className="text-xs text-gray-600">
            Post • <span className="font-mono">{postGroup.postid}</span>
          </p>

          {/* Time - Latest report time */}
          <div className="flex items-center gap-1 text-xs text-gray-500 mt-1">
            <Clock className="w-3 h-3" />
            <span>{formatDate(latestReport.createdAt)}</span>
          </div>
        </div>
      </div>
    </button>
  );
};

export default ReportCard;