#!/usr/bin/env python3
"""
Script to test VideoMAE violence detection model on local videos and images
Model: parksy1314/videomae-base-finetuned-kinetics-finetuned-violence-subset

Supports both video files (.mp4, .avi, .mov, etc.) and image files (.jpg, .png, etc.)
"""

import torch
import numpy as np
from transformers import VideoMAEImageProcessor, VideoMAEForVideoClassification
from pathlib import Path
import cv2
from typing import List
import argparse
import os


class VideoMAEViolenceDetector:
    def __init__(self, model_name: str = "parksy1314/videomae-base-finetuned-kinetics-finetuned-violence-subset"):
        """
        Initialize the VideoMAE model for violence detection.
        
        Args:
            model_name: HuggingFace model identifier
        """
        print(f"Loading model: {model_name}")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Load processor and model
        self.processor = VideoMAEImageProcessor.from_pretrained(model_name)
        self.model = VideoMAEForVideoClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        
        print("Model loaded successfully!")
        print(f"Model config: {self.model.config}")
        
    def load_video(self, video_path: str, num_frames: int = 16) -> np.ndarray:
        """
        Load video and extract frames.
        
        Args:
            video_path: Path to video file
            num_frames: Number of frames to extract
            
        Returns:
            numpy array of video frames
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        print(f"Video info: {total_frames} frames, {fps:.2f} FPS")
        
        # Calculate frame indices to sample
        if total_frames < num_frames:
            indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        else:
            indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        
        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                # Convert BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
        
        cap.release()
        
        if len(frames) == 0:
            raise ValueError(f"Could not extract frames from video: {video_path}")
        
        return np.array(frames)
    
    def load_image(self, image_path: str, num_frames: int = 16) -> np.ndarray:
        """
        Load a single image and replicate it to match video input format.
        
        Args:
            image_path: Path to image file
            num_frames: Number of frames to replicate (default: 16)
            
        Returns:
            numpy array of replicated image frames
        """
        # Load image
        image = cv2.imread(image_path)
        
        if image is None:
            raise ValueError(f"Could not open image: {image_path}")
        
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        print(f"Image info: {image.shape}")
        
        # Replicate the image num_frames times
        frames = np.array([image] * num_frames)
        
        return frames
    
    def predict(self, file_path: str, is_image: bool = False) -> dict:
        """
        Run inference on a video or image file.
        
        Args:
            file_path: Path to video or image file
            is_image: Whether the file is an image (default: False)
            
        Returns:
            Dictionary with prediction results
        """
        print(f"\nProcessing {'image' if is_image else 'video'}: {file_path}")
        
        # Load frames
        if is_image:
            frames = self.load_image(file_path)
        else:
            frames = self.load_video(file_path)
        print(f"Extracted {len(frames)} frames")
        
        # Preprocess frames
        inputs = self.processor(list(frames), return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Run inference
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            
        # Get predictions
        probs = torch.nn.functional.softmax(logits, dim=-1)
        predicted_class_idx = logits.argmax(-1).item()
        confidence = probs[0][predicted_class_idx].item()
        
        # Get label names
        id2label = self.model.config.id2label
        predicted_label = id2label[predicted_class_idx]
        
        # Get all class probabilities
        all_probs = {id2label[i]: probs[0][i].item() for i in range(len(id2label))}
        
        return {
            "predicted_class": predicted_label,
            "predicted_class_idx": predicted_class_idx,
            "confidence": confidence,
            "all_probabilities": all_probs,
            "file_path": file_path,
            "file_type": "image" if is_image else "video"
        }
    
    def predict_batch(self, file_paths: List[str]) -> List[dict]:
        """
        Run inference on multiple videos and/or images.
        
        Args:
            file_paths: List of paths to video/image files
            
        Returns:
            List of prediction results
        """
        results = []
        for file_path in file_paths:
            try:
                # Determine if file is an image or video
                is_image = self._is_image_file(file_path)
                result = self.predict(file_path, is_image=is_image)
                results.append(result)
            except Exception as e:
                print(f"Error processing {file_path}: {str(e)}")
                results.append({
                    "file_path": file_path,
                    "error": str(e)
                })
        return results
    
    def _is_image_file(self, file_path: str) -> bool:
        """Check if file is an image based on extension."""
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp', '.tiff', '.tif'}
        return Path(file_path).suffix.lower() in image_extensions


def get_media_files(folder_path: str, 
                    video_extensions: tuple = ('.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv'),
                    image_extensions: tuple = ('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp', '.tiff', '.tif')) -> List[str]:
    """
    Get all video and image files from a folder.
    
    Args:
        folder_path: Path to folder containing media files
        video_extensions: Tuple of video file extensions to look for
        image_extensions: Tuple of image file extensions to look for
        
    Returns:
        List of media file paths
    """
    media_files = []
    folder = Path(folder_path)
    
    if not folder.exists():
        raise ValueError(f"Folder does not exist: {folder_path}")
    
    if not folder.is_dir():
        raise ValueError(f"Path is not a directory: {folder_path}")
    
    # Search for video files
    for ext in video_extensions:
        media_files.extend(folder.glob(f"*{ext}"))
        media_files.extend(folder.glob(f"*{ext.upper()}"))
    
    # Search for image files
    for ext in image_extensions:
        media_files.extend(folder.glob(f"*{ext}"))
        media_files.extend(folder.glob(f"*{ext.upper()}"))
    
    # Convert to strings and remove duplicates
    media_files = sorted(list(set([str(f) for f in media_files])))
    
    return media_files


def print_results(results: dict):
    """Pretty print the prediction results."""
    print("\n" + "="*60)
    print("PREDICTION RESULTS")
    print("="*60)
    print(f"File: {Path(results['file_path']).name}")
    print(f"Type: {results['file_type']}")
    print(f"Full path: {results['file_path']}")
    print(f"\nPredicted Class: {results['predicted_class']}")
    print(f"Confidence: {results['confidence']:.4f} ({results['confidence']*100:.2f}%)")
    print("\nAll Class Probabilities:")
    for label, prob in sorted(results['all_probabilities'].items(), key=lambda x: x[1], reverse=True):
        print(f"  {label}: {prob:.4f} ({prob*100:.2f}%)")
    print("="*60)


def print_summary(all_results: List[dict]):
    """Print summary of all processed media files."""
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    successful = [r for r in all_results if 'error' not in r]
    failed = [r for r in all_results if 'error' in r]
    
    videos = [r for r in successful if r.get('file_type') == 'video']
    images = [r for r in successful if r.get('file_type') == 'image']
    
    print(f"Total files processed: {len(all_results)}")
    print(f"  Videos: {len(videos)}")
    print(f"  Images: {len(images)}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")
    
    if successful:
        print("\nSuccessful predictions:")
        for result in successful:
            file_type = result.get('file_type', 'unknown')
            print(f"  [{file_type}] {Path(result['file_path']).name}: {result['predicted_class']} ({result['confidence']*100:.2f}%)")
    
    if failed:
        print("\nFailed files:")
        for result in failed:
            print(f"  - {Path(result['file_path']).name}: {result['error']}")
    
    print("="*60)


def main():
    # ============================================================
    # EDIT THIS: Put your video folder path here
    # ============================================================
    VIDEO_FOLDER = "/Users/apple/Downloads/New Moderation/Backend/Videos"
    # ============================================================
    
    parser = argparse.ArgumentParser(description="Test VideoMAE violence detection model on videos and images in a folder")
    parser.add_argument("--folder", type=str, help="Path to folder containing videos/images")
    parser.add_argument("--files", nargs="*", help="Path(s) to individual video/image file(s)")
    parser.add_argument("--model", default="parksy1314/videomae-base-finetuned-kinetics-finetuned-violence-subset",
                        help="HuggingFace model identifier")
    parser.add_argument("--num-frames", type=int, default=16,
                        help="Number of frames to extract from each video")
    parser.add_argument("--video-extensions", nargs="+", default=['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv'],
                        help="Video file extensions to search for")
    parser.add_argument("--image-extensions", nargs="+", default=['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp', '.tiff', '.tif'],
                        help="Image file extensions to search for")
    
    args = parser.parse_args()
    
    # Determine which files to process
    files_to_process = []
    
    if args.files:
        # Process individual files
        files_to_process = args.files
    elif args.folder:
        # Process folder
        try:
            files_to_process = get_media_files(args.folder, 
                                               tuple(args.video_extensions),
                                               tuple(args.image_extensions))
            print(f"Found {len(files_to_process)} media files in {args.folder}")
        except ValueError as e:
            print(f"Error: {e}")
            return
    elif VIDEO_FOLDER and VIDEO_FOLDER != "/path/to/your/video/folder":
        # Use hardcoded folder
        try:
            files_to_process = get_media_files(VIDEO_FOLDER,
                                               tuple(args.video_extensions),
                                               tuple(args.image_extensions))
            print(f"Found {len(files_to_process)} media files in {VIDEO_FOLDER}")
        except ValueError as e:
            print(f"Error: {e}")
            return
    
    if not files_to_process:
        print("Error: No files specified!")
        print("Either:")
        print("  1. Edit VIDEO_FOLDER in the script")
        print("  2. Run with: python test_videomae_violence.py --folder /path/to/media")
        print("  3. Run with: python test_videomae_violence.py --files video1.mp4 image1.jpg")
        return
    
    # Initialize detector
    detector = VideoMAEViolenceDetector(model_name=args.model)
    
    # Process files
    all_results = []
    for i, file_path in enumerate(files_to_process, 1):
        print(f"\n[{i}/{len(files_to_process)}] Processing: {Path(file_path).name}")
        
        if not Path(file_path).exists():
            print(f"Warning: File not found: {file_path}")
            all_results.append({
                "file_path": file_path,
                "error": "File not found"
            })
            continue
        
        try:
            # Determine if file is an image or video
            is_image = detector._is_image_file(file_path)
            results = detector.predict(file_path, is_image=is_image)
            print_results(results)
            all_results.append(results)
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
            import traceback
            traceback.print_exc()
            all_results.append({
                "file_path": file_path,
                "error": str(e)
            })
    
    # Print summary
    if len(all_results) > 1:
        print_summary(all_results)


if __name__ == "__main__":
    main()
    
    
    