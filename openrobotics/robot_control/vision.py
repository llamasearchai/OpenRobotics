"""
Vision processing module for robotics applications.
"""

import logging
from typing import Dict, List, Tuple, Optional, Any, Union
import numpy as np

from openrobotics.robot_control.sensors import Sensor
from openrobotics.mlx_integration.models import VisionModel

logger = logging.getLogger(__name__)

class RobotVision:
    """Vision processing for robot cameras."""
    
    def __init__(
        self,
        camera_sensor: Sensor,
        vision_model: Optional[VisionModel] = None,
        cache_frames: int = 10
    ):
        """
        Initialize robot vision.
        
        Args:
            camera_sensor: Camera sensor to process
            vision_model: Optional vision model for advanced processing
            cache_frames: Number of frames to cache
        """
        if camera_sensor.type != "camera":
            raise ValueError(f"Expected camera sensor, got {camera_sensor.type}")
        
        self.camera = camera_sensor
        self.vision_model = vision_model
        self.cache_frames = cache_frames
        self.frame_cache = []
    
    def get_current_frame(self) -> np.ndarray:
        """
        Get current camera frame.
        
        Returns:
            Current camera frame as numpy array
        """
        camera_data = self.camera.data
        if camera_data is None or "image" not in camera_data:
            raise ValueError("No camera image available")
        
        image = camera_data["image"]
        
        # Cache frame
        self.frame_cache.append(image)
        if len(self.frame_cache) > self.cache_frames:
            self.frame_cache.pop(0)
        
        return image
    
    def detect_objects(self) -> List[Dict[str, Any]]:
        """
        Detect objects in the current frame.
        
        Returns:
            List of detected objects with bounding boxes and labels
        """
        if self.vision_model is None:
            raise ValueError("No vision model available for object detection")
        
        try:
            image = self.get_current_frame()
            
            # Ensure image has correct shape (H, W, C)
            if len(image.shape) != 3 or image.shape[2] != 3:
                if len(image.shape) == 2:  # Grayscale
                    image = np.stack([image] * 3, axis=2)
                else:
                    raise ValueError(f"Unexpected image shape: {image.shape}")
            
            # Process with vision model
            if self.vision_model.config.get("model_type") == "detection":
                # Detection model returns bounding boxes and classes
                result = self.vision_model.predict_with_preprocessing(image)
                
                # Parse results
                if isinstance(result, tuple) and len(result) == 2:
                    boxes, class_scores = result
                    
                    # Convert to list of objects
                    objects = []
                    for i in range(len(boxes)):
                        box = boxes[i]
                        scores = class_scores[i]
                        
                        # Get class with highest score
                        class_id = np.argmax(scores)
                        confidence = scores[class_id]
                        
                        if confidence > 0.3:  # Confidence threshold
                            objects.append({
                                "box": box.tolist(),
                                "class_id": int(class_id),
                                "confidence": float(confidence),
                                "label": self._get_class_name(class_id)
                            })
                    
                    return objects
                else:
                    logger.warning(f"Unexpected detection result format: {type(result)}")
                    return []
            else:
                logger.warning(f"Vision model type '{self.vision_model.config.get('model_type')}' not supported for object detection")
                return []
        except Exception as e:
            logger.error(f"Error detecting objects: {str(e)}")
            return []
    
    def _get_class_name(self, class_id: int) -> str:
        """Get class name from class ID."""
        # Try to get class names from model config
        class_names = self.vision_model.config.get("class_names", [])
        if class_names and class_id < len(class_names):
            return class_names[class_id]
        
        # Default to generic "object_N"
        return f"object_{class_id}"
    
    def detect_motion(self, threshold: float = 30.0) -> bool:
        """
        Detect motion between frames.
        
        Args:
            threshold: Motion detection threshold
            
        Returns:
            True if motion detected
        """
        if len(self.frame_cache) < 2:
            return False
        
        # Get last two frames
        prev_frame = self.frame_cache[-2]
        curr_frame = self.frame_cache[-1]
        
        # Convert to grayscale if needed
        if len(prev_frame.shape) == 3:
            prev_gray = np.mean(prev_frame, axis=2).astype(np.uint8)
        else:
            prev_gray = prev_frame
            
        if len(curr_frame.shape) == 3:
            curr_gray = np.mean(curr_frame, axis=2).astype(np.uint8)
        else:
            curr_gray = curr_frame
        
        # Compute absolute difference
        frame_diff = np.abs(curr_gray.astype(float) - prev_gray.astype(float))
        
        # Check if difference exceeds threshold
        return np.mean(frame_diff) > threshold
    
    def get_distance_to_object(self, object_box: List[float]) -> float:
        """
        Estimate distance to object based on bounding box size.
        
        Args:
            object_box: Bounding box [x1, y1, x2, y2]
            
        Returns:
            Estimated distance in meters
        """
        # Simple distance estimation based on box height
        image_height = self.camera.data.get("image").shape[0]
        box_height = object_box[3] - object_box[1]
        
        # Assuming a linear relationship between inverse height and distance
        # This is a simplified model and would need calibration for accuracy
        normalized_height = box_height / image_height
        
        # Magic numbers that would need calibration in real-world
        if normalized_height > 0:
            distance = 5.0 / normalized_height
            return min(100.0, max(0.1, distance))  # Clamp to reasonable range
        else:
            return 100.0  # Very far away 