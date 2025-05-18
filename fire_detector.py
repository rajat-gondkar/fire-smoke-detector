import cv2
import numpy as np
from sklearn.cluster import KMeans
import os
from pathlib import Path

class FireDetector:
    def __init__(self):
        # Define color ranges for fire and smoke detection
        self.fire_lower = np.array([0, 50, 50])  # HSV lower bound for fire
        self.fire_upper = np.array([20, 255, 255])  # HSV upper bound for fire
        self.smoke_lower = np.array([0, 0, 100])  # HSV lower bound for smoke
        self.smoke_upper = np.array([180, 30, 255])  # HSV upper bound for smoke
        
        # Initialize training data paths
        self.training_path = Path('data/training')
        self.fire_path = self.training_path / 'fire'
        self.smoke_path = self.training_path / 'smoke'
        self.normal_path = self.training_path / 'normal'
        
        # Create directories if they don't exist
        self.fire_path.mkdir(parents=True, exist_ok=True)
        self.smoke_path.mkdir(parents=True, exist_ok=True)
        self.normal_path.mkdir(parents=True, exist_ok=True)
        
        # Load training data
        self.fire_features = self.load_training_data(self.fire_path)
        self.smoke_features = self.load_training_data(self.smoke_path)
        
    def load_training_data(self, path):
        features = []
        for img_file in path.glob('*.jpg'):
            img = cv2.imread(str(img_file))
            if img is not None:
                hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                # Extract color histogram features
                hist = cv2.calcHist([hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])
                cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
                features.append(hist)
        return features
    
    def compare_histograms(self, hist1, hist2):
        return cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
    
    def detect_fire(self, frame):
        # Convert frame to HSV color space
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Create masks for fire and smoke
        fire_mask = cv2.inRange(hsv, self.fire_lower, self.fire_upper)
        smoke_mask = cv2.inRange(hsv, self.smoke_lower, self.smoke_upper)
        
        # Apply morphological operations to reduce noise
        kernel = np.ones((5,5), np.uint8)
        fire_mask = cv2.morphologyEx(fire_mask, cv2.MORPH_OPEN, kernel)
        fire_mask = cv2.morphologyEx(fire_mask, cv2.MORPH_CLOSE, kernel)
        smoke_mask = cv2.morphologyEx(smoke_mask, cv2.MORPH_OPEN, kernel)
        smoke_mask = cv2.morphologyEx(smoke_mask, cv2.MORPH_CLOSE, kernel)
        
        # Find contours for fire and smoke
        fire_contours, _ = cv2.findContours(fire_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        smoke_contours, _ = cv2.findContours(smoke_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Process each contour
        for contour in fire_contours:
            if cv2.contourArea(contour) > 500:  # Filter small contours
                x, y, w, h = cv2.boundingRect(contour)
                roi = frame[y:y+h, x:x+w]
                
                # Extract features from ROI
                roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                roi_hist = cv2.calcHist([roi_hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])
                cv2.normalize(roi_hist, roi_hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
                
                # Compare with training data
                is_fire = any(self.compare_histograms(roi_hist, fire_hist) > 0.7 for fire_hist in self.fire_features)
                
                if is_fire:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
                    cv2.putText(frame, 'Fire', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        
        for contour in smoke_contours:
            if cv2.contourArea(contour) > 500:  # Filter small contours
                x, y, w, h = cv2.boundingRect(contour)
                roi = frame[y:y+h, x:x+w]
                
                # Extract features from ROI
                roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                roi_hist = cv2.calcHist([roi_hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])
                cv2.normalize(roi_hist, roi_hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
                
                # Compare with training data
                is_smoke = any(self.compare_histograms(roi_hist, smoke_hist) > 0.7 for smoke_hist in self.smoke_features)
                
                if is_smoke:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (128, 128, 128), 2)
                    cv2.putText(frame, 'Smoke', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (128, 128, 128), 2)
        
        return frame

def save_training_image(frame, category):
    """Save a frame as a training image in the specified category."""
    detector = FireDetector()
    timestamp = cv2.getTickCount()
    filename = f"{timestamp}.jpg"
    
    if category == 'fire':
        save_path = detector.fire_path / filename
    elif category == 'smoke':
        save_path = detector.smoke_path / filename
    else:
        save_path = detector.normal_path / filename
    
    cv2.imwrite(str(save_path), frame)
    print(f"Saved training image to {save_path}")

def main():
    # Initialize the detector
    detector = FireDetector()
    
    # Open video capture (0 for webcam, or provide video file path)
    video_path = input("Enter the path to your video file (or press Enter for webcam): ").strip()
    if video_path:
        cap = cv2.VideoCapture(video_path)
    else:
        cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open video source")
        return
    
    print("\nControls:")
    print("Press 'f' to save current frame as fire training image")
    print("Press 's' to save current frame as smoke training image")
    print("Press 'n' to save current frame as normal training image")
    print("Press 'q' to quit")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Process the frame
        processed_frame = detector.detect_fire(frame)
        
        # Display the result
        cv2.imshow('Fire and Smoke Detection', processed_frame)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('f'):
            save_training_image(frame, 'fire')
        elif key == ord('s'):
            save_training_image(frame, 'smoke')
        elif key == ord('n'):
            save_training_image(frame, 'normal')
    
    # Clean up
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 