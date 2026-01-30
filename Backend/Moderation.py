#!/usr/bin/env python3
"""
Merged Content Moderation System
- Video/Image Violence Detection (VideoMAE)
- Text Hate Speech Detection
"""

import os
import sys

# FIX MACOS THREADING ISSUES
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

import numpy as np
from transformers import (
    VideoMAEImageProcessor, 
    VideoMAEForVideoClassification,
    AutoTokenizer,
    AutoModelForSequenceClassification
)
from pathlib import Path
import cv2
from typing import List, Dict
from PIL import Image

# Try to import YOLOv8
try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False
    print("Warning: Ultralytics not available. NSFW detection will be disabled.")


# NSFW class definitions for YOLOv8
NUDENET_CLASSES = [
    "BUTTOCKS_EXPOSED",
    "FEMALE_BREAST_EXPOSED",
    "FEMALE_GENITALIA_EXPOSED",
    "ANUS_EXPOSED",
    "MALE_GENITALIA_EXPOSED",
    "BUTTOCKS_COVERED",
]

# ID to label mapping for NSFW classification
ID2LABEL = {
    0: "drawings",
    1: "hentai",
    2: "neutral",
    3: "porn",
    4: "sexy"
}


class VideoMAEViolenceDetector:
    """Violence detection for videos and images using VideoMAE"""
    
    def __init__(self, model_name: str = "parksy1314/videomae-base-finetuned-kinetics-finetuned-violence-subset"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = VideoMAEImageProcessor.from_pretrained(model_name)
        self.model = VideoMAEForVideoClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        
    def load_video(self, video_path: str, num_frames: int = 16) -> np.ndarray:
        """Load video and extract frames"""
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        indices = np.linspace(0, max(0, total_frames - 1), num_frames, dtype=int)
        
        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
        
        cap.release()
        
        if len(frames) == 0:
            raise ValueError(f"Could not extract frames from video: {video_path}")
        
        return np.array(frames)
    
    def load_image(self, image_path: str, num_frames: int = 16) -> np.ndarray:
        """Load a single image and replicate it"""
        image = cv2.imread(image_path)
        
        if image is None:
            raise ValueError(f"Could not open image: {image_path}")
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        frames = np.array([image] * num_frames)
        
        return frames
    
    def predict(self, file_path: str, is_image: bool = False) -> Dict:
        """Run violence detection on video or image"""
        if is_image:
            frames = self.load_image(file_path)
        else:
            frames = self.load_video(file_path)
        
        inputs = self.processor(list(frames), return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            
        probs = torch.nn.functional.softmax(logits, dim=-1)
        predicted_class_idx = logits.argmax(-1).item()
        confidence = probs[0][predicted_class_idx].item()
        
        id2label = self.model.config.id2label
        predicted_label = id2label[predicted_class_idx]
        
        return {
            "predicted_class": predicted_label,
            "confidence": confidence,
            "predicted_class_idx": predicted_class_idx, 
            "file_type": "image" if is_image else "video"
        }
    
    def _is_image_file(self, file_path: str) -> bool:
        """Check if file is an image based on extension"""
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp', '.tiff', '.tif'}
        return Path(file_path).suffix.lower() in image_extensions


class NSFWDetector:
    """Two-stage NSFW detection: Level 1 (transformer) + Level 2 (YOLOv8)"""
    
    def __init__(self, 
                 level1_model_path: str = "./nsfw_model",
                 level2_model_path: str = "./640m.pt"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Level 1: Transformer-based NSFW classifier
        self.level1_model = None
        self.feature_extractor = None
        self.level1_available = False
        
        try:
            from transformers import AutoModelForImageClassification, AutoFeatureExtractor
            if os.path.exists(level1_model_path):
                self.level1_model = AutoModelForImageClassification.from_pretrained(level1_model_path)
                self.feature_extractor = AutoFeatureExtractor.from_pretrained(level1_model_path)
                self.level1_model.to(self.device)
                self.level1_model.eval()
                self.level1_available = True
        except Exception as e:
            print(f"Warning: Level 1 NSFW model not available: {e}")
        
        # Level 2: YOLOv8 NudeNet detector
        self.level2_model = None
        self.level2_model_path = level2_model_path
        self.level2_available = ULTRALYTICS_AVAILABLE and os.path.exists(level2_model_path)
        
        # Classes that confirm NSFW
        self.nsfw_classes = [
            'MALE_GENITALIA_EXPOSED',
            'FEMALE_GENITALIA_EXPOSED',
            'FEMALE_BREAST_EXPOSED',
            'BUTTOCKS_EXPOSED',
            'ANUS_EXPOSED'
        ]
    
    def _load_level2_model(self):
        """Lazy load Level 2 YOLOv8 model only when needed"""
        if self.level2_model is None and self.level2_available:
            try:
                print(f"Loading YOLOv8 Level 2 model from: {self.level2_model_path}")
                self.level2_model = YOLO(self.level2_model_path)
                print("YOLOv8 Level 2 model loaded successfully")
            except Exception as e:
                print(f"Warning: Level 2 NSFW model failed to load: {e}")
                self.level2_available = False
    
    def _predict_level1(self, image_path: str) -> Dict:
        """
        Level 1 NSFW detection - fast initial screening
        Returns: tuple of (safety_rating, confidence)
        """
        if not self.level1_available:
            return {'rating': 'unknown', 'confidence': 0.0}
        
        try:
            # Load and convert image to RGB
            image = Image.open(image_path).convert('RGB')
            
            # Process image
            inputs = self.feature_extractor(images=image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get prediction
            with torch.no_grad():
                outputs = self.level1_model(**inputs)
                scores = torch.nn.functional.softmax(outputs.logits, dim=1)
            
            # Get predicted class ID
            pred_id = torch.argmax(scores, dim=1).item()
            
            # Map predicted class ID to label
            pred_label = ID2LABEL.get(pred_id, "Unknown")
            
            # Get confidence for the predicted class
            confidence = scores[0][pred_id].item()
            
            # Check conditions for NSFW classification (matching original logic)
            if pred_label == "porn":
                return {'rating': 'nsfw', 'confidence': confidence, 'category': pred_label}
            elif pred_label == "neutral":
                return {'rating': 'safe', 'confidence': confidence, 'category': pred_label}
            elif pred_label == "sexy" and confidence >= 0.95:
                return {'rating': 'nsfw', 'confidence': confidence, 'category': pred_label}
            elif pred_label == "hentai" and confidence <= 0.60:
                return {'rating': 'safe', 'confidence': confidence, 'category': pred_label}
            else:
                return {'rating': 'safe', 'confidence': confidence, 'category': pred_label}
        
        except Exception as e:
            return {'rating': 'error', 'confidence': 0.0, 'error': str(e)}
    
    def _detect_level2(self, image_path: str, conf_threshold: float = 0.3) -> List[Dict]:
        """
        Level 2 NSFW detection using YOLOv8 on a single image
        Only detects classes defined in NUDENET_CLASSES
        """
        if not self.level2_available:
            return []
        
        # Lazy load model
        self._load_level2_model()
        
        if self.level2_model is None:
            return []
        
        try:
            # Run inference
            results = self.level2_model(image_path, conf=conf_threshold, verbose=False)
            
            detections = []
            
            for result in results:
                if result.boxes is not None:
                    # Get detection data
                    boxes = result.boxes.xyxy.cpu().numpy()  # x1, y1, x2, y2
                    confidences = result.boxes.conf.cpu().numpy()
                    class_ids = result.boxes.cls.cpu().numpy().astype(int)
                    
                    # Get class names
                    class_names = result.names if hasattr(result, 'names') else {}
                    
                    for box, conf, class_id in zip(boxes, confidences, class_ids):
                        if conf >= conf_threshold:
                            # Get class name
                            class_name = class_names.get(class_id, f"class_{class_id}")
                            
                            # Only include classes in NUDENET_CLASSES (matching original)
                            if class_name in NUDENET_CLASSES:
                                detection = {
                                    'class': class_name,
                                    'score': float(conf),
                                    'box': [float(x) for x in box]
                                }
                                detections.append(detection)
            
            return detections
        
        except Exception as e:
            print(f"Error in Level 2 detection: {str(e)}")
            return []
    
    def predict(self, image_path: str) -> Dict:
        """
        Two-stage NSFW detection with comprehensive logic:
        - Always run both Level 1 and Level 2
        - Level 2 can override Level 1 in both directions
        """
        result = {
            'media_type': 'image',
            'level1': None,
            'level2': None,
            'final_rating': 'safe'
        }
        
        # Stage 1: Fast screening (always run)
        level1_result = self._predict_level1(image_path)
        result['level1'] = level1_result
        
        # Stage 2: Detailed detection (always run for images)
        print(f"Running Level 2 detection...")
        level2_detections = self._detect_level2(image_path)
        result['level2'] = level2_detections
        
        # Check if Level 2 detected any NSFW content
        level2_nsfw_detected = any(
            d['class'] in self.nsfw_classes and d['score'] > 0.3 
            for d in level2_detections
        )
        
        # Decision logic based on both levels
        if level1_result['rating'] == 'nsfw':
            if level2_nsfw_detected:
                result['final_rating'] = 'nsfw'  # Level 2 confirms NSFW
                print("Final decision: Level 1 = NSFW, Level 2 confirms → NSFW")
            else:
                result['final_rating'] = 'safe'  # Level 2 overrides to SAFE
                print("Final decision: Level 1 = NSFW, Level 2 doesn't confirm → SAFE")
        else:  # level1_result['rating'] == 'safe'
            if level2_nsfw_detected:
                result['final_rating'] = 'nsfw'  # Level 2 overrides to NSFW
                print("Final decision: Level 1 = SAFE, Level 2 detects NSFW → NSFW")
            else:
                result['final_rating'] = 'safe'  # Both agree on SAFE
                print("Final decision: Level 1 = SAFE, Level 2 agrees → SAFE")
        
        return result
    
    def _extract_video_frames(self, video_path: str, num_frames: int = 10) -> List[np.ndarray]:
        """Extract frames from video for NSFW analysis"""
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            return []
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames == 0:
            cap.release()
            return []
        
        # Sample frames evenly throughout the video
        indices = np.linspace(0, max(0, total_frames - 1), num_frames, dtype=int)
        
        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                # Convert BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
        
        cap.release()
        return frames
    
    def _predict_level1_on_frame(self, frame: np.ndarray) -> Dict:
        """Run Level 1 prediction on a single frame (numpy array)"""
        if not self.level1_available:
            return {'rating': 'unknown', 'confidence': 0.0}
        
        try:
            # Convert numpy array to PIL Image
            image = Image.fromarray(frame)
            
            # Process image
            inputs = self.feature_extractor(images=image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get prediction
            with torch.no_grad():
                outputs = self.level1_model(**inputs)
                scores = torch.nn.functional.softmax(outputs.logits, dim=1)
            
            # Get predicted class ID
            pred_id = torch.argmax(scores, dim=1).item()
            pred_label = ID2LABEL.get(pred_id, "Unknown")
            confidence = scores[0][pred_id].item()
            
            # Check conditions for NSFW classification
            if pred_label == "porn":
                return {'rating': 'nsfw', 'confidence': confidence, 'category': pred_label}
            elif pred_label == "neutral":
                return {'rating': 'safe', 'confidence': confidence, 'category': pred_label}
            elif pred_label == "sexy" and confidence >= 0.95:
                return {'rating': 'nsfw', 'confidence': confidence, 'category': pred_label}
            elif pred_label == "hentai" and confidence <= 0.60:
                return {'rating': 'safe', 'confidence': confidence, 'category': pred_label}
            else:
                return {'rating': 'safe', 'confidence': confidence, 'category': pred_label}
        
        except Exception as e:
            return {'rating': 'error', 'confidence': 0.0, 'error': str(e)}
    
    def _detect_level2_on_frame(self, frame: np.ndarray, conf_threshold: float = 0.3) -> List[Dict]:
        """Run Level 2 detection on a single frame (numpy array)"""
        if not self.level2_available:
            return []
        
        self._load_level2_model()
        
        if self.level2_model is None:
            return []
        
        try:
            # Convert numpy array to PIL Image for YOLO
            image = Image.fromarray(frame)
            
            # Run inference
            results = self.level2_model(image, conf=conf_threshold, verbose=False)
            
            detections = []
            
            for result in results:
                if result.boxes is not None:
                    boxes = result.boxes.xyxy.cpu().numpy()
                    confidences = result.boxes.conf.cpu().numpy()
                    class_ids = result.boxes.cls.cpu().numpy().astype(int)
                    class_names = result.names if hasattr(result, 'names') else {}
                    
                    for box, conf, class_id in zip(boxes, confidences, class_ids):
                        if conf >= conf_threshold:
                            class_name = class_names.get(class_id, f"class_{class_id}")
                            
                            if class_name in NUDENET_CLASSES:
                                detection = {
                                    'class': class_name,
                                    'score': float(conf),
                                    'box': [float(x) for x in box]
                                }
                                detections.append(detection)
            
            return detections
        
        except Exception as e:
            return []
    
    def predict_video(self, video_path: str, num_frames: int = 10) -> Dict:
        """
        NSFW detection for videos by analyzing sampled frames
        Returns aggregated results across all frames
        """
        result = {
            'media_type': 'video',
            'frames_analyzed': 0,
            'nsfw_frames': 0,
            'level1_results': [],
            'level2_detections': [],
            'final_rating': 'safe'
        }
        
        # Extract frames from video
        frames = self._extract_video_frames(video_path, num_frames)
        
        if not frames:
            result['error'] = 'Could not extract frames from video'
            return result
        
        result['frames_analyzed'] = len(frames)
        
        # Stage 1: Run Level 1 on all frames
        nsfw_frame_indices = []
        for i, frame in enumerate(frames):
            level1_result = self._predict_level1_on_frame(frame)
            result['level1_results'].append({
                'frame_index': i,
                'result': level1_result
            })
            
            if level1_result['rating'] == 'nsfw':
                nsfw_frame_indices.append(i)
                result['nsfw_frames'] += 1
        
        # Stage 2: Run Level 2 only on NSFW frames
        if nsfw_frame_indices:
            print(f"Level 1 flagged {len(nsfw_frame_indices)} frames as NSFW, running Level 2...")
            
            level2_confirmed_frames = 0
            for frame_idx in nsfw_frame_indices:
                frame = frames[frame_idx]
                level2_detections = self._detect_level2_on_frame(frame)
                
                if level2_detections:
                    # Check if any detection confirms NSFW
                    confirmed = any(
                        d['class'] in self.nsfw_classes and d['score'] > 0.3 
                        for d in level2_detections
                    )
                    
                    if confirmed:
                        level2_confirmed_frames += 1
                        result['level2_detections'].append({
                            'frame_index': frame_idx,
                            'detections': level2_detections
                        })
            
            # Video is NSFW if any frame is confirmed NSFW by Level 2
            result['final_rating'] = 'nsfw' if level2_confirmed_frames > 0 else 'safe'
            
            print(f"Level 2 confirmed NSFW in {level2_confirmed_frames} frames")
        else:
            result['final_rating'] = 'safe'
        
        return result


class HateSpeechDetector:
    """Hate speech detection for text content with subcategory classification"""
    
    def __init__(self, 
                 primary_model: str = "IMSyPP/hate_speech_en",
                 subcategory_model: str = "./checkpoint-819"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load primary hate speech model
        self.tokenizer = AutoTokenizer.from_pretrained(primary_model)
        self.model = AutoModelForSequenceClassification.from_pretrained(primary_model)
        self.model.to(self.device)
        self.model.eval()
        
        self.label_map = {
            0: "acceptable",
            1: "inappropriate", 
            2: "offensive",
            3: "violent"
        }
        
        # Load subcategory model (only if hate speech detected)
        self.subcategory_model = None
        self.subcategory_tokenizer = None
        self.subcategory_model_path = subcategory_model
        self.subcategory_labels = {
            0: "non-hate",
            1: "community",
            2: "racial_ethnic",
            3: "religious",
            4: "nationality"
        }
    
    def _load_subcategory_model(self):
        """Lazy load subcategory model only when needed"""
        if self.subcategory_model is None:
            if os.path.exists(self.subcategory_model_path):
                print(f"Loading subcategory model from: {self.subcategory_model_path}")
                self.subcategory_tokenizer = AutoTokenizer.from_pretrained(self.subcategory_model_path)
                self.subcategory_model = AutoModelForSequenceClassification.from_pretrained(self.subcategory_model_path)
                self.subcategory_model.to(self.device)
                self.subcategory_model.eval()
            else:
                print(f"Warning: Subcategory model not found at {self.subcategory_model_path}")
    
    def _classify_subcategory(self, text: str) -> Dict:
        """Classify hate speech into subcategories"""
        self._load_subcategory_model()
        
        if self.subcategory_model is None:
            return None
        
        try:
            inputs = self.subcategory_tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=512
            )
            inputs = inputs.to(self.device)
            
            with torch.no_grad():
                outputs = self.subcategory_model(**inputs)
                logits = outputs.logits
                probabilities = torch.nn.functional.softmax(logits, dim=-1)
            
            predicted_label = logits.argmax().item()
            confidence = probabilities.max().item()
            
            all_probs = {
                self.subcategory_labels[i]: round(probabilities[0][i].item(), 4) 
                for i in range(len(self.subcategory_labels))
            }
            
            return {
                "subcategory": self.subcategory_labels[predicted_label],
                "subcategory_confidence": round(confidence, 3),
                "all_subcategories": all_probs
            }
        except Exception as e:
            return {"error": str(e)}
    
    def predict(self, text: str) -> Dict:
        """Classify hate speech in text with subcategory if hate speech detected"""
        if not text or not text.strip():
            return {
                "label": 0,
                "label_name": "acceptable",
                "confidence": 1.0
            }
        
        try:
            # Primary hate speech detection
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=512
            )
            inputs = inputs.to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
                
            predicted_class_idx = predictions.argmax().item()
            confidence = predictions.max().item()
            label_name = self.label_map.get(predicted_class_idx, "unknown")
            
            result = {
                "label": predicted_class_idx,
                "label_name": label_name,
                "confidence": round(confidence, 3)
            }
            
            # If hate speech detected (inappropriate, offensive, or violent), classify subcategory
            if predicted_class_idx > 0:  # Not "acceptable"
                subcategory_result = self._classify_subcategory(text)
                if subcategory_result:
                    result.update(subcategory_result)
            
            return result
            
        except Exception as e:
            return {
                "label": 0,
                "label_name": "acceptable",
                "confidence": 0.0,
                "error": str(e)
            }


class ContentModerationSystem:
    """Combined moderation system for visual and text content"""
    
    def __init__(self, 
                 subcategory_model_path: str = "/Users/apple/Downloads/New Moderation/Backend/Community/checkpoint-819",
                 nsfw_level1_path: str = "./nsfw_model",
                 nsfw_level2_path: str = "./640m.pt"):
        print("Loading models...")
        self.violence_detector = VideoMAEViolenceDetector()
        self.hate_speech_detector = HateSpeechDetector(
            subcategory_model=subcategory_model_path
        )
        self.nsfw_detector = NSFWDetector(
            level1_model_path=nsfw_level1_path,
            level2_model_path=nsfw_level2_path
        )
        print("Models loaded successfully!\n")
    
    def moderate_content(self, media_path: str, caption: str = None) -> Dict:
        """
        Moderate both visual content and associated text caption
        
        Args:
            media_path: Path to video or image file
            caption: Optional text caption to moderate
            
        Returns:
            Dictionary with moderation results
        """
        results = {
            "media_path": media_path,
            "media_name": Path(media_path).name
        }
        
        # Determine if it's an image or video
        is_image = self.violence_detector._is_image_file(media_path)
        
        # Violence detection (always run)
        try:
            visual_result = self.violence_detector.predict(media_path, is_image=is_image)
            results["violence"] = visual_result
        except Exception as e:
            results["violence"] = {"error": str(e)}
        
        # NSFW detection (for both images and videos)
        try:
            if is_image:
                nsfw_result = self.nsfw_detector.predict(media_path)
            else:
                # Video - use predict_video method
                nsfw_result = self.nsfw_detector.predict_video(media_path)
            results["nsfw"] = nsfw_result
        except Exception as e:
            results["nsfw"] = {"error": str(e)}
        
        # Text content moderation
        if caption:
            text_result = self.hate_speech_detector.predict(caption)
            results["text"] = text_result
            results["caption"] = caption
        else:
            results["text"] = None
            results["caption"] = None
        
        return results
    
    def print_results(self, results: Dict):
        """Print moderation results in a clean format"""
        print("=" * 70)
        print(f"File: {results['media_name']}")
        print("=" * 70)
        
        # Violence detection results
        if "violence" in results:
            violence = results["violence"]
            if "error" in violence:
                print(f"Violence Detection: ERROR - {violence['error']}")
            else:
                print(f"Violence Detection:")
                print(f"  Type: {violence['file_type']}")
                print(f"  Class: {violence['predicted_class']}")
                print(f"  Confidence: {violence['confidence']:.2%}")
                print(f" class index: {violence['predicted_class_idx']}")
        
        # NSFW detection results (for both images and videos)
        if results.get("nsfw"):
            nsfw = results["nsfw"]
            if "error" in nsfw:
                print(f"\nNSFW Detection: ERROR - {nsfw['error']}")
            else:
                media_type = nsfw.get('media_type', 'unknown')
                print(f"\nNSFW Detection ({media_type.upper()}):")
                
                if media_type == 'video':
                    # Video NSFW results
                    print(f"  Frames Analyzed: {nsfw.get('frames_analyzed', 0)}")
                    print(f"  NSFW Frames Flagged: {nsfw.get('nsfw_frames', 0)}")
                    
                    if nsfw.get('level2_detections'):
                        print(f"\n  Level 2 Confirmed Detections:")
                        for frame_det in nsfw['level2_detections']:
                            frame_idx = frame_det['frame_index']
                            detections = frame_det['detections']
                            print(f"    Frame {frame_idx}: {len(detections)} detection(s)")
                            for det in detections:
                                print(f"      - {det['class']}: {det['score']:.2%}")
                    
                    print(f"\n  Final Rating: {nsfw.get('final_rating', 'unknown').upper()}")
                
                elif media_type == 'image':
                    # Image NSFW results
                    if nsfw.get("level1"):
                        level1 = nsfw["level1"]
                        print(f"  Level 1 (Fast Screening):")
                        print(f"    Category: {level1.get('category', 'N/A')}")
                        print(f"    Rating: {level1.get('rating', 'N/A')}")
                        print(f"    Confidence: {level1.get('confidence', 0):.2%}")
                    
                    if nsfw.get("level2"):
                        level2 = nsfw["level2"]
                        if level2:
                            print(f"\n  Level 2 (Detailed Detection):")
                            print(f"    Detections found: {len(level2)}")
                            for i, detection in enumerate(level2, 1):
                                print(f"    [{i}] {detection['class']}: {detection['score']:.2%}")
                    
                    print(f"\n  Final Rating: {nsfw.get('final_rating', 'unknown').upper()}")
        
        # Text moderation results
        if results.get("text"):
            text = results["text"]
            print(f"\nText Analysis:")
            print(f"  Caption: '{results['caption']}'")
            if "error" in text:
                print(f"  ERROR: {text['error']}")
            else:
                print(f"  Primary Label: {text['label_name']}")
                print(f"  Primary Confidence: {text['confidence']:.2%}")
                
                # Show subcategory if hate speech detected
                if "subcategory" in text:
                    print(f"\n  Hate Speech Subcategory:")
                    print(f"    Type: {text['subcategory']}")
                    print(f"    Confidence: {text['subcategory_confidence']:.2%}")
                    
                    if "all_subcategories" in text:
                        print(f"\n  All Subcategory Probabilities:")
                        for label, prob in sorted(text['all_subcategories'].items(), 
                                                  key=lambda x: x[1], reverse=True):
                            print(f"    {label}: {prob:.2%}")
        
        print("=" * 70 + "\n")


def main():
    # ============================================================
    # CONFIGURATION
    # ============================================================
    # Path to subcategory hate speech model
    SUBCATEGORY_MODEL_PATH = "/Users/apple/Downloads/New Moderation/Backend/Community/checkpoint-819"
    
    # Paths to NSFW detection models
    NSFW_LEVEL1_MODEL_PATH = "./nsfw_model"  # Transformer-based fast screening
    NSFW_LEVEL2_MODEL_PATH = "./640m.pt"     # YOLOv8 detailed detection
    
    # ============================================================
    # TEST CASES - Edit these paths and captions
    # ============================================================
    test_cases = [
        {
            "media_path": "/Users/apple/Downloads/IMG_0346.HEIC",
            "caption": "what a pussy"
        },
        {
            "media_path": "/Users/apple/Downloads/New Moderation/Backend/Videos/architecturementality_1718442496_3390863183454087913_2103441796.mp4",
            "caption": "Can't stand these coming into our neighborhood"
        },
        {
            "media_path": "/Users/apple/Downloads/New Moderation/Backend/lafdavlog/gorevideosonly_1760455617_3743295366407199108_60511355591.mp4",
            "caption": "Kill all american niggers" # No caption for this one
        }
    ]
    # ============================================================
    
    # Initialize moderation system
    moderator = ContentModerationSystem(
        subcategory_model_path=SUBCATEGORY_MODEL_PATH,
        nsfw_level1_path=NSFW_LEVEL1_MODEL_PATH,
        nsfw_level2_path=NSFW_LEVEL2_MODEL_PATH
    )
    
    # Run moderation on each test case
    for i, test_case in enumerate(test_cases, 1):
        media_path = test_case["media_path"]
        caption = test_case.get("caption")
        
        print(f"\n[Test Case {i}/{len(test_cases)}]")
        
        # Validate media file exists
        if not Path(media_path).exists():
            print(f"Error: File not found: {media_path}")
            print("=" * 70 + "\n")
            continue
        
        # Run moderation
        results = moderator.moderate_content(media_path, caption)
        
        # Display results
        moderator.print_results(results)


if __name__ == "__main__":
    main()