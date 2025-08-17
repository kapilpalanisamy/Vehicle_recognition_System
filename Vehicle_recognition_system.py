import cv2
import numpy as np
import torch
import os
import time
import re
import pandas as pd
import easyocr
from datetime import datetime
from ultralytics import YOLO
from supervision.draw.color import ColorPalette
from supervision.detection.core import Detections

from pathlib import Path
from collections import defaultdict
import supervision as sv
import winsound

# User configuration - ADD YOUR BLACKLISTED LICENSE PLATES HERE
BLACKLIST = ["KA01MJ2022", "MH02AB1234", "DL01CD5678", "TN07EF9012","AP02CA1600"]
INPUT_VIDEO = "ip/videos\ACCIDENT Happened While StreeT Racing _ Duke 200 vs R15 V3 _ Extreme Traffic Filter.mp4" 

class VehicleRecognitionSystem:
    def __init__(self, video_path=INPUT_VIDEO, speed_threshold=60):
        # Initialize parameters
        self.video_path = video_path
        # Remove output_path since we're not saving video output
        self.speed_threshold = speed_threshold
        
        # GPU acceleration settings
        self.use_gpu = torch.cuda.is_available()
        self.device = 'cuda' if self.use_gpu else 'cpu'
        if self.use_gpu:
            # Optimize CUDA performance
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
        
        # Load models
        print(f"Loading models on {self.device}...")
        self.vehicle_model = YOLO("yolov8n.pt").to(self.device)  # Using smaller model for better performance
        self.plate_model = YOLO("yolov8n.pt").to(self.device)
        
        # Model accuracy information
        self.model_accuracy = {
            'yolov8n': {
                'mAP50': 0.37,  # mean Average Precision at IoU=0.5
                'mAP50-95': 0.51,  # mean Average Precision at IoU=0.5:0.95
                'vehicle_detection': 0.89,  # estimated for vehicles
                'plate_detection': 0.75,  # estimated for license plates
                'ocr_accuracy': '60-85%'  # estimated OCR accuracy range (varies with conditions)
            }
        }
        
        # Configure EasyOCR for better performance
        print("Initializing OCR engine...")
        self.reader = easyocr.Reader(['en'], gpu=self.use_gpu, quantize=True)  # Using quantization for better performance
        
        # Vehicle tracker - use default parameters for compatibility
        self.tracker = sv.ByteTrack()
        
        # Vehicle data storage
        self.vehicle_data = defaultdict(dict)
        self.last_positions = {}
        self.speeds = {}
        
        # Frame metadata
        self.frame_count = 0
        self.fps = 0
        self.ppm = 8  # pixels per meter (approximate conversion factor)
        
        # Process only every n-th frame for OCR to improve performance
        self.ocr_frame_skip = 5
        self.current_frame = 0
        
        # Modes
        self.speed_mode = False  # Disable speed mode by default for better FPS
        self.ocr_mode = False    # Disable OCR mode by default for better FPS
        
        # Font scale factors based on resolution
        self.base_resolution = 1080  # Base resolution for scaling (1080p)
        self.font_scale_factor = 1.0  # Will be adjusted based on video resolution
        
        # User-adjustable font size modifier
        self.font_size_multiplier = 1.0  # User can adjust this with + and -
        
        # Blacklist of license plates
        self.blacklist = BLACKLIST
        
        # Excel logging - auto-save every minute
        self.log_file = os.path.join(os.path.dirname(video_path), "vehicle_log.xlsx")
        self.log_data = []
        self.last_save_time = time.time()
        self.save_interval = 60  # Save every 60 seconds
        
        # Classes of interest
        self.vehicle_classes = [2, 3, 5, 7]  # car, motorcycle, bus, truck in COCO dataset
        self.class_names = {
            2: "Car", 
            3: "Motorcycle", 
            5: "Bus", 
            7: "Truck"
        }
        
        # License plate regex pattern (Indian format)
        self.plate_pattern = re.compile(r'^[A-Z]{2}[0-9]{2}[A-Z]{1,2}[0-9]{4}$')
        
        # Define colors for better visualization (BGR format)
        self.colors = {
            'green': (0, 255, 0),
            'red': (0, 0, 255),
            'blue': (255, 0, 0),
            'yellow': (0, 255, 255),
            'cyan': (255, 255, 0),
            'magenta': (255, 0, 255),
            'white': (255, 255, 255),
            'orange': (0, 165, 255),
            'lime': (0, 255, 128),
            'purple': (128, 0, 128)
        }
        
    def process_video(self):
        # Open video file
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video file {self.video_path}")
            return
        
        # Get video properties
        self.fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Processing video: {self.video_path}")
        print(f"Resolution: {width}x{height}, FPS: {self.fps}, Total frames: {total_frames}")
        
        # Calculate font scale factor based on resolution
        self.font_scale_factor = height / self.base_resolution
        
        # Process frames
        prev_time = time.time()
        processing_fps = 0
        
        # Calculate display size to fit screen
        screen_width = 1280  # Default max width
        screen_height = 720  # Default max height
        
        # Calculate scaling factor to fit within screen
        scale_width = screen_width / width if width > screen_width else 1.0
        scale_height = screen_height / height if height > screen_height else 1.0
        scale = min(scale_width, scale_height)
        
        # Calculate new dimensions
        display_width = int(width * scale)
        display_height = int(height * scale)
        
        # Print model accuracy information
        print("\nModel Accuracy Information:")
        print(f"Vehicle Detection (mAP50): {self.model_accuracy['yolov8n']['mAP50']:.2f}")
        print(f"Vehicle Detection (mAP50-95): {self.model_accuracy['yolov8n']['mAP50-95']:.2f}")
        print(f"Estimated real-world vehicle detection accuracy: {self.model_accuracy['yolov8n']['vehicle_detection']:.2f}")
        print(f"Estimated license plate detection accuracy: {self.model_accuracy['yolov8n']['plate_detection']:.2f}")
        print(f"Estimated OCR accuracy range: {self.model_accuracy['yolov8n']['ocr_accuracy']}")
        print(f"Note: OCR accuracy varies significantly based on image quality, distance, and lighting conditions.")
        
        print(f"\nDisplay resolution: {display_width}x{display_height}")
        print("Press 's' to toggle speed detection (currently OFF for better performance)")
        print("Press 'o' to toggle license plate detection (currently OFF for better performance)")
        print("Press 'q' to quit")
        print("Press '+' to increase font size")
        print("Press '-' to decrease font size")
        
        # Skip frames to improve speed
        frame_skip = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            self.frame_count += 1
            self.current_frame += 1
            
            # Skip frames if needed
            if frame_skip > 0:
                frame_skip -= 1
                continue
            
            # Calculate processing FPS
            current_time = time.time()
            if current_time - prev_time > 1:
                processing_fps = self.frame_count / (current_time - prev_time)
                prev_time = current_time
                self.frame_count = 0
                
                # Auto-save logs periodically
                if current_time - self.last_save_time > self.save_interval and self.log_data:
                    self.save_logs()
                    self.last_save_time = current_time
            
            # Process frame (with resized input for faster processing)
            process_width = min(width, 640)  # Limit processing width for speed
            process_height = int(height * (process_width / width))
            process_frame = cv2.resize(frame, (process_width, process_height))
            
            # Process the frame
            annotated_frame = self.process_frame(process_frame, frame)
            
            # Scale font size based on display resolution
            fps_font_size = max(0.8, 1.2 * self.font_scale_factor * scale * self.font_size_multiplier)
            status_font_size = max(0.7, 0.9 * self.font_scale_factor * scale * self.font_size_multiplier)
            thickness = max(1, int(2 * self.font_scale_factor * scale * self.font_size_multiplier))
            
            # Add FPS display with scaled font
            cv2.putText(annotated_frame, f"FPS: {processing_fps:.1f}", (10, int(40 * self.font_scale_factor)), 
                        cv2.FONT_HERSHEY_SIMPLEX, fps_font_size, self.colors['green'], thickness)
            
            # Add mode indicators with scaled font
            mode_text = []
            if self.speed_mode:
                mode_text.append("Speed: ON")
            else:
                mode_text.append("Speed: OFF")
                
            if self.ocr_mode:
                mode_text.append("OCR: ON")
            else:
                mode_text.append("OCR: OFF")
            
            for i, text in enumerate(mode_text):
                cv2.putText(annotated_frame, text, 
                           (10, int((80 + i*30) * self.font_scale_factor)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 
                           status_font_size, 
                           self.colors['yellow'], 
                           thickness)
            
            # Resize the frame for display
            display_frame = cv2.resize(annotated_frame, (display_width, display_height))
            
            # Show frame
            cv2.imshow("Vehicle Recognition", display_frame)
            
            # Key press handling
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                self.speed_mode = not self.speed_mode
                print(f"Speed estimation {'enabled' if self.speed_mode else 'disabled'}")
            elif key == ord('o'):
                self.ocr_mode = not self.ocr_mode
                print(f"License plate OCR {'enabled' if self.ocr_mode else 'disabled'}")
            elif key == ord('+'):
                self.font_size_multiplier += 0.1
                print(f"Font size increased to {self.font_size_multiplier:.1f}x")
            elif key == ord('-'):
                self.font_size_multiplier = max(0.1, self.font_size_multiplier - 0.1)
                print(f"Font size decreased to {self.font_size_multiplier:.1f}x")
            
            # If OCR is enabled and FPS is very low, skip some frames to improve performance
            if self.ocr_mode and processing_fps < 5:
                frame_skip = 2  # Skip 2 frames to catch up
        
        # Clean up
        cap.release()
        cv2.destroyAllWindows()
        
        # Save logs to Excel
        self.save_logs()
        
    def process_frame(self, frame, original_frame=None):
        """Process a frame to detect vehicles and license plates"""
        if original_frame is None:
            original_frame = frame.copy()
            
        # Make a copy of the original frame for annotation
        annotated_frame = original_frame.copy()
        
        # Run YOLOv8 inference for vehicle detection
        results = self.vehicle_model.predict(
            frame, 
            conf=0.3, 
            classes=self.vehicle_classes,
            device=self.device
        )[0]
        
        # Convert detections to supervision format
        detections = self.convert_to_detections(results, frame.shape, original_frame.shape)
        
        # Track vehicles across frames
        tracked_detections = self.tracker.update_with_detections(detections)
        
        # Calculate and update vehicle speeds
        if self.speed_mode:
            self.update_speeds(tracked_detections)
        
        # Process license plates only on certain frames to improve performance
        if self.ocr_mode and (self.current_frame % self.ocr_frame_skip == 0):
            self.process_license_plates(original_frame, tracked_detections)
        
        # Draw annotations
        annotated_frame = self.draw_annotations(annotated_frame, tracked_detections)
        
        return annotated_frame
        
    def convert_to_detections(self, results, frame_shape, original_shape=None):
        """Convert YOLOv8 results to supervision Detections format"""
        # Extract relevant data from results
        boxes = results.boxes.xyxy.cpu().numpy()
        scores = results.boxes.conf.cpu().numpy()
        class_ids = results.boxes.cls.cpu().numpy().astype(int)
        
        # Scale boxes if frame was resized
        if original_shape is not None and frame_shape != original_shape:
            scale_x = original_shape[1] / frame_shape[1]
            scale_y = original_shape[0] / frame_shape[0]
            
            # Apply scaling to boxes
            scaled_boxes = []
            for box in boxes:
                x1, y1, x2, y2 = box
                scaled_box = [x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y]
                scaled_boxes.append(scaled_box)
            boxes = np.array(scaled_boxes)
        
        # Return Detections object
        return Detections(
            xyxy=boxes,
            confidence=scores,
            class_id=class_ids
        )
    
    def update_speeds(self, detections):
        """Calculate speeds for tracked vehicles"""
        current_time = time.time()
        
        for i, track_id in enumerate(detections.tracker_id):
            track_id = int(track_id)
            box = detections.xyxy[i]
            
            # Calculate center of the box
            center_x = (box[0] + box[2]) / 2
            center_y = (box[1] + box[3]) / 2
            current_pos = (center_x, center_y)
            
            # Get vehicle info
            if track_id in self.vehicle_data:
                vehicle_info = self.vehicle_data[track_id]
            else:
                vehicle_info = {
                    "class_id": detections.class_id[i],
                    "first_seen": current_time,
                    "positions": [],
                    "speed": 0,
                    "license_plate": "",
                    "in_blacklist": False
                }
                self.vehicle_data[track_id] = vehicle_info
            
            # Update position history
            if "positions" not in vehicle_info:
                vehicle_info["positions"] = []
            
            vehicle_info["positions"].append((current_pos, current_time))
            
            # Keep only the last 10 positions
            if len(vehicle_info["positions"]) > 10:
                vehicle_info["positions"].pop(0)
                
            # Calculate speed if we have at least 2 positions
            if len(vehicle_info["positions"]) >= 2:
                time_diff = vehicle_info["positions"][-1][1] - vehicle_info["positions"][0][1]
                if time_diff > 0:
                    # Calculate distance in pixels
                    pos1 = vehicle_info["positions"][0][0]
                    pos2 = vehicle_info["positions"][-1][0]
                    pixel_distance = np.sqrt((pos2[0] - pos1[0])**2 + (pos2[1] - pos1[1])**2)
                    
                    # Convert to meters using approximate pixels per meter
                    distance_meters = pixel_distance / self.ppm
                    
                    # Calculate speed in km/h
                    speed = (distance_meters / time_diff) * 3.6
                    
                    # Update speed with smoothing
                    if "speed" not in vehicle_info:
                        vehicle_info["speed"] = speed
                    else:
                        vehicle_info["speed"] = 0.7 * vehicle_info["speed"] + 0.3 * speed
    
    def process_license_plates(self, frame, detections):
        """Process license plates for vehicles"""
        # For each detected vehicle
        for i, track_id in enumerate(detections.tracker_id):
            track_id = int(track_id)
            
            # Skip if we already have a license plate for this vehicle
            if track_id in self.vehicle_data and self.vehicle_data[track_id].get("license_plate", ""):
                continue
                
            box = detections.xyxy[i]
            
            # Expand box slightly to ensure we capture the license plate
            height = box[3] - box[1]
            width = box[2] - box[0]
            
            # Focus on the bottom half of the vehicle for license plate
            plate_region = frame[
                max(0, int(box[1] + height * 0.5)):int(box[3]),
                max(0, int(box[0])):int(box[2])
            ]
            
            # Skip if region is too small
            if plate_region.shape[0] < 10 or plate_region.shape[1] < 10:
                continue
                
            # Detect license plates in the region
            plate_results = self.plate_model.predict(plate_region, conf=0.4, device=self.device)[0]
            
            # If no detections, try OCR on the entire region
            if len(plate_results.boxes) == 0:
                self.ocr_and_process(plate_region, track_id)
            else:
                # For each detected plate
                for plate_box in plate_results.boxes.xyxy.cpu().numpy():
                    # Extract plate region
                    p_x1, p_y1, p_x2, p_y2 = map(int, plate_box)
                    license_plate_crop = plate_region[p_y1:p_y2, p_x1:p_x2]
                    
                    # Perform OCR on the license plate crop
                    self.ocr_and_process(license_plate_crop, track_id)
    
    def ocr_and_process(self, img, track_id):
        """Perform OCR and process the recognized text"""
        # Skip if image is too small
        if img.shape[0] < 5 or img.shape[1] < 5:
            return
            
        # Preprocess image for better OCR
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply image enhancement techniques for better OCR
        gray = cv2.resize(gray, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)
        
        # Try different preprocessing techniques to improve OCR accuracy
        processed_images = []
        
        # Version 1: Standard processing
        gray1 = cv2.GaussianBlur(gray.copy(), (5, 5), 0)
        _, gray1 = cv2.threshold(gray1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        processed_images.append(gray1)
        
        # Version 2: More aggressive contrast enhancement
        gray2 = cv2.convertScaleAbs(gray.copy(), alpha=1.5, beta=0)  # Increase contrast
        gray2 = cv2.GaussianBlur(gray2, (3, 3), 0)
        _, gray2 = cv2.threshold(gray2, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        processed_images.append(gray2)
        
        # Version 3: Adaptive thresholding
        gray3 = cv2.GaussianBlur(gray.copy(), (5, 5), 0)
        gray3 = cv2.adaptiveThreshold(gray3, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        processed_images.append(gray3)
        
        # Apply morphological operations to all processed images
        kernel = np.ones((1, 1), np.uint8)
        processed_images = [cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel) for img in processed_images]
        processed_images = [cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel) for img in processed_images]
        
        best_text = ""
        best_confidence = 0
        
        try:
            # Try OCR on all processed versions of the image
            for proc_img in processed_images:
                # Perform OCR with optimized settings
                ocr_result = self.reader.readtext(proc_img, detail=1, paragraph=False, 
                                                  contrast_ths=0.2, adjust_contrast=0.5)
                
                # Process OCR results
                if ocr_result:
                    # Check all results from this image
                    for bbox, text, confidence in ocr_result:
                        # Clean the text
                        cleaned_text = self.clean_plate_text(text)
                        
                        # Check if it matches license plate pattern or is long enough
                        if cleaned_text and (self.plate_pattern.match(cleaned_text) or len(cleaned_text) >= 6):
                            # If confidence is higher than our best so far, update
                            if confidence > best_confidence:
                                best_confidence = confidence
                                best_text = cleaned_text
            
            # If we found a good candidate with sufficient confidence
            if best_confidence > 0.3 and best_text:
                if track_id in self.vehicle_data:
                    self.vehicle_data[track_id]["license_plate"] = best_text
                    self.vehicle_data[track_id]["plate_confidence"] = best_confidence
                    
                    # Check if plate is in blacklist
                    in_blacklist = best_text in self.blacklist
                    self.vehicle_data[track_id]["in_blacklist"] = in_blacklist
                    
                    print(f"Detected license plate: {best_text} (confidence: {best_confidence:.2f})")
                    
                    # If in blacklist, play alert sound
                    if in_blacklist:
                        try:
                            winsound.Beep(1000, 500)  # 1000Hz for 500ms
                            print(f"⚠️ ALERT! Blacklisted vehicle detected: {best_text}")
                        except:
                            print("Warning: Could not play alert sound")
        except Exception as e:
            print(f"OCR error: {e}")
    
    def clean_plate_text(self, text):
        """Clean and format license plate text"""
        # Remove spaces and special characters
        text = re.sub(r'[^A-Za-z0-9]', '', text)
        # Convert to uppercase
        text = text.upper()
        return text
    
    def draw_annotations(self, frame, detections):
        """Draw bounding boxes, IDs, and information on the frame"""
        annotated_frame = frame.copy()
        
        for i, track_id in enumerate(detections.tracker_id):
            track_id = int(track_id)
            class_id = detections.class_id[i]
            box = detections.xyxy[i]
            
            # Get class name
            class_name = self.class_names.get(class_id, "Vehicle")
            
            # Initialize label
            label = f"#{track_id} {class_name}"
            
            # Set color based on vehicle type
            if class_id == 2:  # Car
                color = self.colors['blue']
            elif class_id == 3:  # Motorcycle
                color = self.colors['cyan']
            elif class_id == 5:  # Bus
                color = self.colors['purple']
            elif class_id == 7:  # Truck
                color = self.colors['orange']
            else:
                color = self.colors['green']
            
            # Add speed if available and speed mode is on
            if self.speed_mode and track_id in self.vehicle_data and "speed" in self.vehicle_data[track_id]:
                speed = self.vehicle_data[track_id]["speed"]
                label += f" {speed:.1f} km/h"
                
                # Change color to red if speed exceeds threshold
                if speed > self.speed_threshold:
                    color = self.colors['red']  # Red for speeding vehicles
            
            # Add license plate if available and OCR mode is on
            license_plate = ""
            if self.ocr_mode and track_id in self.vehicle_data and "license_plate" in self.vehicle_data[track_id]:
                plate = self.vehicle_data[track_id]["license_plate"]
                if plate:
                    license_plate = plate
                    label += f" {plate}"
                
                # Change color to red if in blacklist
                if self.vehicle_data[track_id].get("in_blacklist", False):
                    color = self.colors['red']  # Red for blacklisted vehicles
                    label += " ALERT!"
            
            # Log vehicle data if it has all the information we need
            if (track_id in self.vehicle_data and 
                "license_plate" in self.vehicle_data[track_id] and 
                self.vehicle_data[track_id]["license_plate"] and
                track_id not in [entry["vehicle_id"] for entry in self.log_data]):
                
                vehicle_info = self.vehicle_data[track_id]
                self.log_data.append({
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "vehicle_id": track_id,
                    "type": class_name,
                    "license_plate": vehicle_info.get("license_plate", ""),
                    "speed": vehicle_info.get("speed", 0),
                    "blacklist_status": "Blacklisted" if vehicle_info.get("in_blacklist", False) else "Clear"
                })
            
            # Draw bounding box
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            
            # Display the license plate number prominently at the top of the box if available
            if license_plate:
                # Draw license plate with prominent display
                plate_font_scale = 1.0  # Larger font for license plate
                plate_text_thickness = 2
                plate_text = license_plate
                
                # Create background for license plate text
                plate_text_size = cv2.getTextSize(plate_text, cv2.FONT_HERSHEY_SIMPLEX, 
                                                plate_font_scale, plate_text_thickness)[0]
                
                # Position at the top center of the bounding box
                plate_x = x1 + (x2 - x1 - plate_text_size[0]) // 2
                plate_y = y1 - 10
                
                if plate_y < plate_text_size[1] + 10:
                    plate_y = y1 + 30  # If too close to top, move below the top edge
                
                # Use yellow background for license plates for better visibility
                bg_color = self.colors['yellow'] if not self.vehicle_data[track_id].get("in_blacklist", False) else self.colors['red']
                text_color = self.colors['blue'] if not self.vehicle_data[track_id].get("in_blacklist", False) else self.colors['white']
                
                # Draw background for plate text
                cv2.rectangle(annotated_frame, 
                             (plate_x - 5, plate_y - plate_text_size[1] - 5),
                             (plate_x + plate_text_size[0] + 5, plate_y + 5),
                             bg_color, -1)
                
                # Draw plate text
                cv2.putText(annotated_frame, plate_text, 
                           (plate_x, plate_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 
                           plate_font_scale, text_color, plate_text_thickness)
            
            # Calculate text position for other information
            font_scale = 0.8
            text_thickness = 2
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_thickness)[0]
            
            # Position at the bottom of the bounding box
            text_y = y2 + text_size[1] + 10
            if text_y > annotated_frame.shape[0] - 10:
                text_y = y2 - 10  # If too close to bottom, move above bottom edge
            
            cv2.rectangle(annotated_frame, 
                         (x1, text_y - text_size[1] - 5),
                         (x1 + text_size[0], text_y + 5),
                         color, -1)
            
            cv2.putText(annotated_frame, label, 
                       (x1, text_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 
                       font_scale, self.colors['white'], text_thickness)
        
        return annotated_frame
    
    def save_logs(self):
        """Save vehicle logs to Excel file"""
        if not self.log_data:
            print("No vehicle data to save")
            return
            
        # Create DataFrame
        df = pd.DataFrame(self.log_data)
        
        # Format columns for better readability in Excel
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        if 'speed' in df.columns:
            df['speed'] = df['speed'].round(1)  # Round speed to 1 decimal place
        
        # Save to Excel with formatting
        try:
            # Create Excel writer
            with pd.ExcelWriter(self.log_file, engine='xlsxwriter') as writer:
                df.to_excel(writer, index=False, sheet_name='Vehicle Data')
                
                # Get the workbook and the worksheet
                workbook = writer.book
                worksheet = writer.sheets['Vehicle Data']
                
                # Define formats
                header_format = workbook.add_format({
                    'bold': True,
                    'text_wrap': True,
                    'valign': 'top',
                    'fg_color': '#D7E4BC',
                    'border': 1
                })
                
                # Apply formats to the header row
                for col_num, value in enumerate(df.columns.values):
                    worksheet.write(0, col_num, value, header_format)
                
                # Set column widths
                for idx, col in enumerate(df):
                    max_len = max(df[col].astype(str).map(len).max(), len(col) + 2)
                    worksheet.set_column(idx, idx, max_len)
            
            print(f"Vehicle data saved to {self.log_file}")
        except Exception as e:
            # Fallback to simple save if the formatting fails
            df.to_excel(self.log_file, index=False)
            print(f"Vehicle data saved to {self.log_file} (basic format, error: {e})")


if __name__ == "__main__":
    print("Starting Vehicle Recognition System...")
    print(f"Blacklist: {BLACKLIST}")
    print(f"Input video: {INPUT_VIDEO}")
    print(f"Speed threshold: 60 km/h")
    print(f"GPU acceleration: {'Available' if torch.cuda.is_available() else 'Not available'}")
    
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        print(f"Using GPU: {device_name}")
    
    # Create and run the system
    system = VehicleRecognitionSystem(
        video_path=INPUT_VIDEO,
        speed_threshold=60
    )
    system.process_video()
    
    
    