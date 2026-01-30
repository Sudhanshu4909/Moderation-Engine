// ReportFilters.jsx - Filter controls for reports

import React from 'react';
import { Filter, Search } from 'lucide-react';

const ReportFilters = ({ filters, onFilterChange }) => {
  return (
    <div className="p-4 border-b border-purple-100/50 bg-white/50 space-y-3">
      {/* Search */}
      <div className="relative">
        <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
          <Search className="w-4 h-4 text-gray-400" />
        </div>
        <input
          type="text"
          placeholder="Search by Post ID, Reporter Email..."
          value={filters.search || ''}
          onChange={(e) => onFilterChange({ ...filters, search: e.target.value })}
          className="w-full pl-10 pr-3 py-2.5 border border-purple-200 rounded-lg text-sm bg-white focus:outline-none focus:ring-2 focus:ring-purple-500 focus:border-transparent"
        />
      </div>

      {/* Filters Row */}
      <div className="grid grid-cols-2 gap-3">
        {/* Report Type Filter */}
        <div>
          <div className="flex items-center gap-2 mb-2">
            <Filter className="w-4 h-4 text-purple-600" />
            <label className="text-xs font-semibold text-gray-900">Report Type</label>
          </div>
          <select
            value={filters.reportType || 'all'}
            onChange={(e) => onFilterChange({ ...filters, reportType: e.target.value })}
            className="w-full px-3 py-2 border border-purple-200 rounded-lg text-sm bg-white focus:outline-none focus:ring-2 focus:ring-purple-500 focus:border-transparent"
          >
            <option value="all">All Types</option>
            <option value="Inappropriate Content">❌ Inappropriate Content</option>
            <option value="Hateful Or Abusive Content">❌ Hateful/Abusive</option>
            <option value="Copyright/Trademark Infringe">⚠️ Copyright</option>
            <option value="Spam or Misleading">⚠️ Spam</option>
            <option value="Content not Visible/Playable">🔧 Technical</option>
            <option value="Other/Content">📝 Other</option>
          </select>
        </div>

        {/* Status Filter */}
        <div>
  <div className="flex items-center gap-2 mb-2">
    <Filter className="w-4 h-4 text-purple-600" />
    <label className="text-xs font-semibold text-gray-900">Status</label>
  </div>
  <select
    value={filters.status || 'all'}
    onChange={(e) => onFilterChange({ ...filters, status: e.target.value })}
    className="w-full px-3 py-2 border border-purple-200 rounded-lg text-sm bg-white focus:outline-none focus:ring-2 focus:ring-purple-500 focus:border-transparent"
  >
    <option value="all">All Status</option>
    <option value="pending">⏳ Pending</option>
    <option value="reviewed">📋 Reviewed</option>
    <option value="resolved">✅ Resolved</option>
    <option value="rejected">❌ Rejected</option>
  </select>
</div>
      </div>

      {/* Sort */}
      <div>
        <div className="flex items-center gap-2 mb-2">
          <Filter className="w-4 h-4 text-purple-600" />
          <label className="text-xs font-semibold text-gray-900">Sort By</label>
        </div>
        <select
          value={filters.sortBy || 'newest'}
          onChange={(e) => onFilterChange({ ...filters, sortBy: e.target.value })}
          className="w-full px-3 py-2 border border-purple-200 rounded-lg text-sm bg-white focus:outline-none focus:ring-2 focus:ring-purple-500 focus:border-transparent"
        >
          <option value="newest">Newest First</option>
          <option value="oldest">Oldest First</option>
        </select>
      </div>

      {/* Active Filters Count */}
      {(filters.reportType !== 'all' || filters.status !== 'all' || filters.search) && (
        <div className="flex items-center justify-between pt-2 border-t border-purple-100">
          <span className="text-xs text-gray-600">
            {[
              filters.reportType !== 'all' && 'Type',
              filters.status !== 'all' && 'Status',
              filters.search && 'Search'
            ].filter(Boolean).join(', ')} active
          </span>
          <button
            onClick={() => onFilterChange({ reportType: 'all', status: 'all', search: '', sortBy: 'newest' })}
            className="text-xs font-semibold text-purple-600 hover:text-purple-700"
          >
            Clear All
          </button>
        </div>
      )}
    </div>
  );
};

export default ReportFilters;