"""
FastAPI Backend for Content Moderation System
Processes content and stores results for frontend display
"""

import os
import sys
import time
# FIX MACOS THREADING ISSUES
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
from pathlib import Path
from datetime import datetime
import json
import shutil

# Import the moderation system
from Moderation import ContentModerationSystem

# Initialize FastAPI app
app = FastAPI(title="Content Moderation API", version="1.0.0")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

VIOLENCE_CLASS_IDS = [1] 
# Configuration
MODELS_CONFIG = {
    "subcategory_model": "/Users/apple/Downloads/New Moderation/Backend/Community/checkpoint-819",
    "nsfw_level1": "./nsfw_model",
    "nsfw_level2": "./640m.pt"
}

# Media storage directory
MEDIA_DIR = Path("./media_files")
MEDIA_DIR.mkdir(exist_ok=True)

# Mount media directory for serving files
app.mount("/media", StaticFiles(directory=str(MEDIA_DIR)), name="media")

# In-memory storage for moderation results
moderation_results = []

# Initialize moderation system (lazy loading)
moderator = None

def get_moderator():
    """Lazy load the moderation system"""
    global moderator
    if moderator is None:
        print("Initializing Content Moderation System...")
        moderator = ContentModerationSystem(
            subcategory_model_path=MODELS_CONFIG["subcategory_model"],
            nsfw_level1_path=MODELS_CONFIG["nsfw_level1"],
            nsfw_level2_path=MODELS_CONFIG["nsfw_level2"]
        )
    return moderator


def copy_media_file(source_path: str, content_id: int) -> str:
    """Copy media file to server directory and return URL"""
    source = Path(source_path)
    if not source.exists():
        raise FileNotFoundError(f"Source file not found: {source_path}")
    
    extension = source.suffix
    timestamp = int(time.time() * 1000)  # ← ADD THIS
    filename = f"content_{content_id}_{timestamp}{extension}"  # ← CHANGE THIS
    dest = MEDIA_DIR / filename
    shutil.copy2(source, dest)
    return f"/media/{filename}"


def map_to_frontend_format(results: Dict, content_id: int, title: str, content_type: str, 
                           duration: str = None, media_url: str = None) -> Dict:
    """
    Map backend moderation results to frontend expected format
    """
    
    # Initialize flags
    flags = {
        "nudity": {"detected": False, "confidence": 0.0},
        "violence": {"detected": False, "confidence": 0.0},
        "hateSpeech": {"detected": False, "confidence": 0.0},
        "communityTargeted": {"detected": False, "confidence": 0.0},
        "political": {"detected": False, "confidence": 0.0}
    }
    
    
    # Map Violence Detection - FIXED LOGIC
    if "violence" in results and not results["violence"].get("error"):
        violence_result = results["violence"]
        confidence = violence_result.get("confidence", 0.0)
        predicted_class = violence_result.get("predicted_class", "")
        predicted_class_idx = violence_result.get("predicted_class_idx")
        
        if predicted_class_idx is not None:
            # Use class ID for accurate detection
            is_violent = predicted_class_idx in VIOLENCE_CLASS_IDS
            flags["violence"]["detected"] = is_violent
            flags["violence"]["confidence"] = confidence
            print(f"   [Violence] ID={predicted_class_idx}, Name='{predicted_class}', Violent={is_violent}, Conf={confidence:.1%}")
        else:
            print(f"violence not working correctly")
    
    # Map NSFW Detection
    if "nsfw" in results and not results["nsfw"].get("error"):
        nsfw = results["nsfw"]
        final_rating = nsfw.get("final_rating", "safe").lower()
        
        # For images
        if nsfw.get("media_type") == "image" and nsfw.get("level1"):
            confidence = nsfw["level1"].get("confidence", 0.0)
            flags["nudity"]["detected"] = final_rating == "nsfw"
            flags["nudity"]["confidence"] = confidence
        
        # For videos
        elif nsfw.get("media_type") == "video":
            if nsfw.get("level1_results"):
                nsfw_confidences = [
                    frame["result"]["confidence"] 
                    for frame in nsfw["level1_results"] 
                    if frame["result"].get("rating") == "nsfw"
                ]
                avg_confidence = sum(nsfw_confidences) / len(nsfw_confidences) if nsfw_confidences else 0.0
                flags["nudity"]["detected"] = final_rating == "nsfw"
                flags["nudity"]["confidence"] = avg_confidence
    
    # Map Text/Caption Analysis
    if "text" in results and results["text"]:
        text = results["text"]
        
        if not text.get("error"):
            label = text.get("label", 0)
            confidence = text.get("confidence", 0.0)
            
            # Map hate speech detection
            is_hate_speech = label > 0
            flags["hateSpeech"]["detected"] = is_hate_speech
            flags["hateSpeech"]["confidence"] = confidence if is_hate_speech else 1.0 - confidence
            
            # Map subcategory to communityTargeted
            if "subcategory" in text:
                subcategory = text["subcategory"]
                print(f"   [Text] Subcategory='{subcategory}', Label={label}, Conf={confidence:.1%}")
                subcategory_confidence = text.get("subcategory_confidence", 0.0)
                
                community_categories = ["community", "racial_ethnic", "religious", "nationality"]
                is_community_targeted = subcategory in community_categories
                
                flags["communityTargeted"]["detected"] = is_community_targeted
                flags["communityTargeted"]["confidence"] = subcategory_confidence if is_community_targeted else 0.0
            
            # Political detection
            caption = results.get("caption", "").lower()
            political_keywords = ["politics", "rally", "election", "government", "policy", "vote"]
            has_political_content = any(keyword in caption for keyword in political_keywords)
            
            if has_political_content and is_hate_speech:
                flags["political"]["detected"] = True
                flags["political"]["confidence"] = confidence * 0.8
    
    # Determine if vertical based on content type
    is_vertical = content_type in ["SNIP", "SSUP"]
    
    return {
        "id": content_id,
        "type": content_type,
        "title": title,
        "duration": duration,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "caption": results.get("caption", ""),
        "flags": flags,
        "isVertical": is_vertical,
        "mediaUrl": media_url,
        "thumbnailUrl": media_url
    }


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "online",
        "service": "Content Moderation API",
        "version": "1.0.0",
        "total_results": len(moderation_results)
    }


@app.get("/api/content")
async def get_all_content():
    """
    Get all moderation results for frontend display
    
    Returns:
        List of all moderated content with flags
    """
    return {
        "success": True,
        "count": len(moderation_results),
        "content": moderation_results
    }


@app.get("/api/content/{content_id}")
async def get_content_by_id(content_id: int):
    """
    Get specific content by ID
    
    Args:
        content_id: ID of the content
        
    Returns:
        Content details with moderation results
    """
    for content in moderation_results:
        if content["id"] == content_id:
            return {
                "success": True,
                "content": content
            }
    
    raise HTTPException(status_code=404, detail="Content not found")


@app.post("/api/process")
async def process_content_from_path(
    media_path: str,
    caption: Optional[str] = None,
    title: Optional[str] = None,
    content_type: str = "SNIP",
    duration: Optional[str] = None
):
    """
    Process content from file path (backend use only)
    
    This endpoint is called by your backend script to process content
    and store results for frontend display
    
    Args:
        media_path: Path to media file on server
        caption: Optional caption text
        title: Content title
        content_type: Type (SNIP/SHOT/MINI/SSUP)
        duration: Video duration
        
    Returns:
        Processed moderation results
    """
    
    # Validate file exists
    if not Path(media_path).exists():
        raise HTTPException(status_code=404, detail=f"File not found: {media_path}")
    
    try:
        # Get moderator instance
        mod = get_moderator()
        
        # Run moderation
        print(f"Processing: {media_path}")
        results = mod.moderate_content(media_path, caption)
        
        # Generate ID
        content_id = len(moderation_results) + 1
        
        # Copy media file
        try:
            media_url = copy_media_file(media_path, content_id)
            print(f"✅ Media copied: {media_url}")
        except Exception as e:
            print(f"⚠️  Could not copy media: {e}")
            media_url = None
        
        # Map to frontend format
        content_data = map_to_frontend_format(
            results, 
            content_id, 
            title or Path(media_path).name,
            content_type,
            duration,
            media_url 
        )
        
        # Store result
        moderation_results.append(content_data)
        
        print(f"✅ Processed content ID {content_id}")
        
        return {
            "success": True,
            "content_id": content_id,
            "message": "Content processed successfully",
            "data": content_data
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")


@app.delete("/api/content/{content_id}")
async def delete_content(content_id: int):
    """
    Remove content from results (after review)
    
    Args:
        content_id: ID of content to remove
        
    Returns:
        Success message
    """
    global moderation_results
    
    for i, content in enumerate(moderation_results):
        if content["id"] == content_id:
            removed = moderation_results.pop(i)
            
            # Optionally delete media file
            if removed.get("mediaUrl"):
                try:
                    media_filename = removed["mediaUrl"].split("/")[-1]
                    media_path = MEDIA_DIR / media_filename
                    if media_path.exists():
                        media_path.unlink()
                        print(f"Deleted media file: {media_filename}")
                except Exception as e:
                    print(f"Warning: Could not delete media file: {e}")
            
            return {
                "success": True,
                "message": f"Content {content_id} removed",
                "removed": removed
            }
    
    raise HTTPException(status_code=404, detail="Content not found")


@app.post("/api/clear")
async def clear_all_results():
    """Clear all moderation results"""
    global moderation_results
    count = len(moderation_results)
    
    # Clear media files
    try:
        for file in MEDIA_DIR.glob("content_*"):
            file.unlink()
        print(f"Cleared media files")
    except Exception as e:
        print(f"Warning: Could not clear media files: {e}")
    
    moderation_results = []
    
    return {
        "success": True,
        "message": f"Cleared {count} results"
    }


@app.get("/api/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "models": {
            "violence": "VideoMAE",
            "nsfw_level1": "Transformer",
            "nsfw_level2": "YOLOv8",
            "hate_speech": "BERT-based",
            "subcategory": "Fine-tuned classifier"
        },
        "loaded": moderator is not None,
        "total_content": len(moderation_results),
        "media_directory": str(MEDIA_DIR)
    }


if __name__ == "__main__":
    import uvicorn
    
    print("=" * 70)
    print("Content Moderation API Server")
    print("=" * 70)
    print("Starting server on http://localhost:8000")
    print("API Documentation: http://localhost:8000/docs")
    print("Frontend fetches from: http://localhost:8000/api/content")
    print(f"Media files served from: http://localhost:8000/media/")
    print("=" * 70)
    
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")