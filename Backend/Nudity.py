import os
import numpy as np
from PIL import Image
import torch
from transformers import AutoModelForImageClassification, AutoFeatureExtractor
from collections import defaultdict

# Import YOLOv8 from ultralytics
try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
    print("Ultralytics YOLO imported successfully")
except ImportError:
    ULTRALYTICS_AVAILABLE = False
    print("Ultralytics not available. Install with: pip install ultralytics")

# Global NudeNet detector
nude_detector = None

# NudeNet class definitions for YOLOv8 - only the classes you want to detect
NUDENET_CLASSES = [
    "BUTTOCKS_EXPOSED",
    "FEMALE_BREAST_EXPOSED",
    "FEMALE_GENITALIA_EXPOSED",
    "ANUS_EXPOSED",
    "MALE_GENITALIA_EXPOSED",
    "BUTTOCKS_COVERED",
]

class YOLOv8NudeDetector:
    def __init__(self, model_path):
        """
        Initialize YOLOv8 NudeNet detector
        
        Args:
            model_path (str): Path to the YOLOv8 .pt model file
        """
        self.model = None
        self.model_path = model_path
        self.load_model()
    
    def load_model(self):
        """Load the YOLOv8 model"""
        try:
            if not ULTRALYTICS_AVAILABLE:
                print("ERROR: Ultralytics not available. Install with: pip install ultralytics")
                return
            
            print(f"Loading YOLOv8 model from: {self.model_path}")
            
            # Check if model file exists
            if not os.path.exists(self.model_path):
                print(f"ERROR: Model file not found at {self.model_path}")
                return
            
            print(f"Model file found. Size: {os.path.getsize(self.model_path)} bytes")
            
            # Load YOLOv8 model
            self.model = YOLO(self.model_path)
            print("YOLOv8 NudeNet model loaded successfully")
            
            # Print model info
            if hasattr(self.model, 'model') and hasattr(self.model.model, 'names'):
                print(f"Model classes: {self.model.model.names}")
            
        except Exception as e:
            print(f"Error loading YOLOv8 model: {str(e)}")
            import traceback
            traceback.print_exc()
            self.model = None
    
    def detect_batch(self, image_paths, conf_threshold=0.3):
        """
        Detect nude content using YOLOv8 on multiple images in batch
        
        Args:
            image_paths (list): List of paths to image files
            conf_threshold (float): Confidence threshold for detections
            
        Returns:
            dict: Detection results for each image path
        """
        if self.model is None:
            print("YOLOv8 model not loaded")
            return {}
        
        if not image_paths:
            return {}
        
        try:
            print(f"Running YOLOv8 batch detection on {len(image_paths)} images")
            
            # Run batch inference - more efficient than individual predictions
            results = self.model(image_paths, conf=conf_threshold, verbose=False)
            
            batch_detections = {}
            
            # Process results for each image
            for i, (image_path, result) in enumerate(zip(image_paths, results)):
                detections = []
                
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
                            
                            # Only include classes in NUDENET_CLASSES
                            if class_name in NUDENET_CLASSES:
                                # Convert box format to [x1, y1, x2, y2]
                                box_coords = [float(x) for x in box]
                                
                                detection = {
                                    'class': class_name,
                                    'score': float(conf),
                                    'box': box_coords
                                }
                                detections.append(detection)
                                
                                print(f"Detection in {image_path}: {class_name} (confidence: {conf:.3f})")
                
                batch_detections[image_path] = detections
            
            total_detections = sum(len(dets) for dets in batch_detections.values())
            print(f"YOLOv8 batch processing completed. Found {total_detections} total detections")
            return batch_detections
            
        except Exception as e:
            print(f"Error during YOLOv8 batch detection: {str(e)}")
            import traceback
            traceback.print_exc()
            return {}

def initialize_nude_detector(model_path):
    """Initialize the YOLOv8 NudeNet detector"""
    global nude_detector
    try:
        print(f"Attempting to initialize YOLOv8 NudeNet detector with model path: {model_path}")
        
        # Check if model file exists
        if not os.path.exists(model_path):
            print(f"ERROR: Model file does not exist at path: {model_path}")
            nude_detector = None
            return
        
        print(f"Model file found. File size: {os.path.getsize(model_path)} bytes")
        
        nude_detector = YOLOv8NudeDetector(model_path)
        
        if nude_detector.model is not None:
            print("YOLOv8 NudeNet detector initialized successfully")
        else:
            print("YOLOv8 detector created but model failed to load")
            nude_detector = None
            
    except Exception as e:
        print(f"Error initializing YOLOv8 NudeNet detector: {str(e)}")
        import traceback
        traceback.print_exc()
        nude_detector = None

def detect_nsfw_level2_batch(image_paths):
    """
    Level 2 NSFW detection using YOLOv8 NudeNet model on multiple images
    Only detects classes defined in NUDENET_CLASSES
    
    Args:
        image_paths: List of paths to image files
    
    Returns:
        dict: Detection results from YOLOv8 filtered by NUDENET_CLASSES for each image
    """
    global nude_detector
    
    if nude_detector is None:
        print("YOLOv8 NudeNet detector not initialized, skipping Level 2 detection")
        return {}
    
    if nude_detector.model is None:
        print("YOLOv8 model not loaded, skipping Level 2 detection")
        return {}
    
    if not image_paths:
        print("No image paths provided for Level 2 detection")
        return {}
    
    try:
        print(f"Running Level 2 NSFW detection on {len(image_paths)} images")
        
        # Use batch detection for efficiency
        batch_results = nude_detector.detect_batch(image_paths, conf_threshold=0.3)
        
        print(f"Level 2 batch detection completed. Processed {len(batch_results)} images")
        return batch_results
        
    except Exception as e:
        print(f"Error in Level 2 NSFW batch detection: {str(e)}")
        import traceback
        traceback.print_exc()
        return {}

def load_image_moderation_model(model_path="nsfw_model"):
    """Load the transformer-based image moderation model"""
    model = AutoModelForImageClassification.from_pretrained(model_path)
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_path)
    return model, feature_extractor

# ID to label mapping for NSFW classification
ID2LABEL = {
    0: "drawings",
    1: "hentai",
    2: "neutral",
    3: "porn",
    4: "sexy"
}

def predict_safety_rating(model_tuple, image_path):
    """
    Level 1 NSFW detection - fast initial screening
    
    Args:
        model_tuple: Tuple of (model, feature_extractor)
        image_path: Path to the image file
    
    Returns:
        tuple: (safety_rating, confidence)
    """
    model, feature_extractor = model_tuple
    
    # Load and convert image to RGB
    image = Image.open(image_path).convert('RGB')
    
    # Process image
    inputs = feature_extractor(images=image, return_tensors="pt")
    
    # Get prediction
    with torch.no_grad():
        outputs = model(**inputs)
        scores = torch.nn.functional.softmax(outputs.logits, dim=1)
    
    # Get predicted class ID
    pred_id = torch.argmax(scores, dim=1).item()
    
    # Map predicted class ID to label using manual ID2LABEL mapping
    pred_label = ID2LABEL.get(pred_id, "Unknown")
    
    # Get confidence for the predicted class
    confidence = scores[0][pred_id].item()

    # Check conditions for NSFW classification
    if pred_label in ["porn"]:
        return 'nsfw', confidence
    elif pred_label == "neutral":
        return 'safe', confidence
    elif pred_label == "sexy" and confidence >= 0.95:
        return 'nsfw', confidence
    elif pred_label == "hentai" and confidence <= 0.60:
        return 'safe', confidence
    else:
        return 'safe', confidence


def analyze_images_for_nsfw(image_paths, level1_model_path, level2_model_path):
    """
    Two-stage NSFW detection system for images
    
    Args:
        image_paths (list): List of paths to image files
        level1_model_path (str): Path to Level 1 transformer model
        level2_model_path (str): Path to Level 2 YOLOv8 model (.pt file)
    
    Returns:
        dict: Results containing safety ratings and detections for each image
    """
    # Initialize models
    print("Loading Level 1 model...")
    level1_model = load_image_moderation_model(level1_model_path)
    
    print("Loading Level 2 model...")
    initialize_nude_detector(level2_model_path)
    
    results = {
        'level1_results': {},
        'level2_results': {},
        'final_ratings': {}
    }
    
    # Stage 1: Run Level 1 detection on all images
    print(f"\n=== Running Level 1 detection on {len(image_paths)} images ===")
    nsfw_images = []
    
    for image_path in image_paths:
        safety_rating, confidence = predict_safety_rating(level1_model, image_path)
        results['level1_results'][image_path] = {
            'rating': safety_rating,
            'confidence': confidence
        }
        
        if safety_rating == 'nsfw':
            nsfw_images.append(image_path)
            print(f"Level 1 - NSFW detected: {image_path} (confidence: {confidence:.3f})")
    
    # Stage 2: Run Level 2 detection only on NSFW images
    if nsfw_images:
        print(f"\n=== Running Level 2 detection on {len(nsfw_images)} NSFW images ===")
        level2_detections = detect_nsfw_level2_batch(nsfw_images)
        results['level2_results'] = level2_detections
        
        # Determine final ratings
        nsfw_classes = [
            'MALE_GENITALIA_EXPOSED',
            'FEMALE_GENITALIA_EXPOSED',
            'FEMALE_BREAST_EXPOSED',
            'BUTTOCKS_EXPOSED',
            'ANUS_EXPOSED'
        ]
        
        for image_path in image_paths:
            level1_rating = results['level1_results'][image_path]['rating']
            
            if level1_rating == 'safe':
                results['final_ratings'][image_path] = 'safe'
            else:
                # Check Level 2 results
                level2_confirmed = False
                if image_path in level2_detections:
                    for detection in level2_detections[image_path]:
                        if detection['class'] in nsfw_classes and detection['score'] > 0.3:
                            level2_confirmed = True
                            break
                
                results['final_ratings'][image_path] = 'nsfw' if level2_confirmed else 'safe'
    else:
        print("\nNo NSFW images detected by Level 1, skipping Level 2")
        for image_path in image_paths:
            results['final_ratings'][image_path] = 'safe'
    
    return results


# Example usage
if __name__ == "__main__":
    # Test with sample images
    test_images = [
        '/Users/apple/Downloads/New Moderation/Backend/Videos/BzYPKF.mp4'
        
    ]
    
    level1_model_path = './nsfw_model'
    level2_model_path = './640m.pt'
    
    # Check if files exist
    existing_images = [img for img in test_images if os.path.exists(img)]
    
    if not existing_images:
        print("No test images found. Please provide valid image paths.")
    else:
        print(f"Testing NSFW detection on {len(existing_images)} images...")
        results = analyze_images_for_nsfw(existing_images, level1_model_path, level2_model_path)
        
        print("\n=== FINAL RESULTS ===")
        for image_path, rating in results['final_ratings'].items():
            print(f"{image_path}: {rating}")
            
            
            