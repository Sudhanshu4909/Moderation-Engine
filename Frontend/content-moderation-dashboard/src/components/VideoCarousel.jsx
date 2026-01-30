// VideoCarousel.jsx
import React, { useState } from 'react';
import { ChevronLeft, ChevronRight } from 'lucide-react';
import VideoPlayer from './VideoPlayer';

const VideoCarousel = ({ videos, contentType }) => {
  const [currentIndex, setCurrentIndex] = useState(0);

  // Safety check: if no videos, return null or placeholder
  if (!videos || videos.length === 0) {
    return (
      <div className="w-full h-64 flex items-center justify-center bg-gray-200 text-gray-500 rounded-lg">
        <div className="text-center">
          <p className="text-sm">No Media Available</p>
        </div>
      </div>
    );
  }

  // Reset index if it's out of bounds (when switching to a report with fewer videos)
  const safeIndex = currentIndex >= videos.length ? 0 : currentIndex;
  const currentMedia = videos[safeIndex];

  const goToPrevious = () => {
    const newIndex = safeIndex === 0 ? videos.length - 1 : safeIndex - 1;
    setCurrentIndex(newIndex);
  };

  const goToNext = () => {
    const newIndex = safeIndex === videos.length - 1 ? 0 : safeIndex + 1;
    setCurrentIndex(newIndex);
  };

  const goToIndex = (index) => {
    setCurrentIndex(index);
  };

  return (
    <div className="relative">
      {/* Main Media Display */}
      <div className="rounded-lg overflow-hidden bg-black">
        <div className="bg-gray-800 px-4 py-2 flex justify-between items-center">
          <p className="text-white text-sm font-medium">
            {currentMedia.type === 'image' ? '🖼️ Image' : '🎥 Video'} {safeIndex + 1} of {videos.length}
            {contentType === 'Interactive SNIP' && ` • ID: ${currentMedia.id}`}
          </p>
          {currentMedia.duration && (
            <span className="text-gray-300 text-xs">
              Duration: {currentMedia.duration}s
            </span>
          )}
        </div>
        
        <div className="relative w-full bg-black" style={{ height: '400px' }}>
          {currentMedia.type === 'image' ? (
            <div className="flex items-center justify-center h-full">
              <img 
                src={currentMedia.url} 
                className="max-h-full w-auto object-contain" 
                alt={`Media ${safeIndex + 1}`}
              />
            </div>
          ) : currentMedia.type === 'video' ? (
            <VideoPlayer 
              key={currentMedia.url}
              src={currentMedia.url} 
              className="w-full h-full"
            />
          ) : (
            <div className="w-full h-full flex items-center justify-center bg-gray-700 text-white">
              <div className="text-center">
                <p className="text-sm">Unknown Media Type</p>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Navigation Buttons */}
      {videos.length > 1 && (
        <>
          <button
            onClick={goToPrevious}
            className="absolute left-4 top-1/2 -translate-y-1/2 bg-black/50 hover:bg-black/70 text-white p-2 rounded-full transition-colors z-20"
            aria-label="Previous"
          >
            <ChevronLeft className="w-6 h-6" />
          </button>
          
          <button
            onClick={goToNext}
            className="absolute right-4 top-1/2 -translate-y-1/2 bg-black/50 hover:bg-black/70 text-white p-2 rounded-full transition-colors z-20"
            aria-label="Next"
          >
            <ChevronRight className="w-6 h-6" />
          </button>
        </>
      )}

      {/* Dots Indicator */}
      {videos.length > 1 && (
        <div className="flex justify-center gap-2 mt-4">
          {videos.map((_, index) => (
            <button
              key={index}
              onClick={() => goToIndex(index)}
              className={`w-2 h-2 rounded-full transition-all ${
                index === safeIndex 
                  ? 'bg-blue-600 w-6' 
                  : 'bg-gray-300 hover:bg-gray-400'
              }`}
              aria-label={`Go to media ${index + 1}`}
            />
          ))}
        </div>
      )}
    </div>
  );
};

export default VideoCarousel;