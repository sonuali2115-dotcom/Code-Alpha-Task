import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict

class SimpleSORT:
    """Simplified SORT tracker for object tracking"""
    def __init__(self, max_age=30, min_hits=3, iou_threshold=0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.tracks = []
        self.track_id_counter = 0
        
    def iou(self, box1, box2):
        """Calculate IoU between two boxes [x1, y1, x2, y2]"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0
    
    def update(self, detections):
        """
        Update tracks with new detections
        detections: list of [x1, y1, x2, y2, confidence, class_id]
        """
        # Update existing tracks
        for track in self.tracks:
            track['age'] += 1
            track['hits'] = 0
        
        # Match detections to tracks
        if len(detections) > 0 and len(self.tracks) > 0:
            iou_matrix = np.zeros((len(detections), len(self.tracks)))
            for d, det in enumerate(detections):
                for t, track in enumerate(self.tracks):
                    iou_matrix[d, t] = self.iou(det[:4], track['bbox'])
            
            # Simple greedy matching
            matched_indices = set()
            for _ in range(min(len(detections), len(self.tracks))):
                max_iou = np.max(iou_matrix)
                if max_iou < self.iou_threshold:
                    break
                    
                d, t = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
                self.tracks[t]['bbox'] = detections[d][:4]
                self.tracks[t]['class_id'] = int(detections[d][5])
                self.tracks[t]['confidence'] = detections[d][4]
                self.tracks[t]['age'] = 0
                self.tracks[t]['hits'] += 1
                self.tracks[t]['total_hits'] += 1
                matched_indices.add(d)
                iou_matrix[d, :] = -1
                iou_matrix[:, t] = -1
            
            # Create new tracks for unmatched detections
            for d, det in enumerate(detections):
                if d not in matched_indices:
                    self.tracks.append({
                        'id': self.track_id_counter,
                        'bbox': det[:4],
                        'class_id': int(det[5]),
                        'confidence': det[4],
                        'age': 0,
                        'hits': 1,
                        'total_hits': 1
                    })
                    self.track_id_counter += 1
        elif len(detections) > 0:
            # No existing tracks, create new ones
            for det in detections:
                self.tracks.append({
                    'id': self.track_id_counter,
                    'bbox': det[:4],
                    'class_id': int(det[5]),
                    'confidence': det[4],
                    'age': 0,
                    'hits': 1,
                    'total_hits': 1
                })
                self.track_id_counter += 1
        
        # Remove old tracks
        self.tracks = [t for t in self.tracks if t['age'] < self.max_age]
        
        # Return confirmed tracks
        return [t for t in self.tracks if t['total_hits'] >= self.min_hits]


class ObjectDetectionTracker:
    def __init__(self, model_name='yolov8n.pt', confidence_threshold=0.5):
        """
        Initialize the object detection and tracking system
        
        Args:
            model_name: YOLO model to use (yolov8n.pt, yolov8s.pt, etc.)
            confidence_threshold: Minimum confidence for detections
        """
        print(f"Loading YOLO model: {model_name}...")
        self.model = YOLO(model_name)
        self.confidence_threshold = confidence_threshold
        self.tracker = SimpleSORT(max_age=30, min_hits=3, iou_threshold=0.3)
        self.track_history = defaultdict(lambda: [])
        self.colors = {}
        
    def get_color(self, track_id):
        """Generate consistent color for each track ID"""
        if track_id not in self.colors:
            np.random.seed(track_id)
            self.colors[track_id] = tuple(map(int, np.random.randint(0, 255, 3)))
        return self.colors[track_id]
    
    def process_frame(self, frame, show_trails=True):
        """
        Process a single frame for detection and tracking
        
        Args:
            frame: Input frame from video
            show_trails: Whether to show tracking trails
            
        Returns:
            Annotated frame with detections and tracks
        """
        # Run YOLO detection
        results = self.model(frame, verbose=False)[0]
        
        # Extract detections
        detections = []
        if results.boxes is not None:
            boxes = results.boxes.xyxy.cpu().numpy()
            confidences = results.boxes.conf.cpu().numpy()
            class_ids = results.boxes.cls.cpu().numpy()
            
            for box, conf, cls_id in zip(boxes, confidences, class_ids):
                if conf >= self.confidence_threshold:
                    detections.append([*box, conf, cls_id])
        
        # Update tracker
        tracks = self.tracker.update(detections)
        
        # Draw detections and tracks
        annotated_frame = frame.copy()
        
        for track in tracks:
            x1, y1, x2, y2 = map(int, track['bbox'])
            track_id = track['id']
            class_id = track['class_id']
            confidence = track['confidence']
            
            # Get class name
            class_name = self.model.names[class_id]
            
            # Get color for this track
            color = self.get_color(track_id)
            
            # Draw bounding box
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw label with track ID
            label = f"ID:{track_id} {class_name} {confidence:.2f}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(annotated_frame, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), color, -1)
            cv2.putText(annotated_frame, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Track center point for trails
            center = ((x1 + x2) // 2, (y1 + y2) // 2)
            self.track_history[track_id].append(center)
            
            # Keep only last 30 points
            if len(self.track_history[track_id]) > 30:
                self.track_history[track_id].pop(0)
            
            # Draw tracking trail
            if show_trails and len(self.track_history[track_id]) > 1:
                points = np.array(self.track_history[track_id], dtype=np.int32)
                cv2.polylines(annotated_frame, [points], False, color, 2)
        
        # Display statistics
        info_text = f"Detected: {len(detections)} | Tracked: {len(tracks)}"
        cv2.putText(annotated_frame, info_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        return annotated_frame
    
    def run_video(self, source=0, show_trails=True, save_output=None):
        """
        Run detection and tracking on video source
        
        Args:
            source: Video source (0 for webcam, or path to video file)
            show_trails: Whether to show tracking trails
            save_output: Path to save output video (optional)
        """
        cap = cv2.VideoCapture(source)
        
        if not cap.isOpened():
            print(f"Error: Could not open video source {source}")
            return
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"Video: {width}x{height} @ {fps}fps")
        
        # Setup video writer if saving output
        writer = None
        frames_to_save = []  # Fallback: store frames in memory
        if save_output:
            # Ensure output has correct extension
            if not save_output.endswith(('.mp4', '.avi')):
                save_output = save_output + '.avi'
            
            print(f"Initializing video writer...")
            
            # Try MJPG codec first (works with Windows Media Player)
            if save_output.endswith('.mp4'):
                save_output = save_output.replace('.mp4', '.avi')
            
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            writer = cv2.VideoWriter(save_output, fourcc, fps, (width, height))
            
            # Test write
            if writer.isOpened():
                print(f"✓ Video writer initialized: {save_output}")
                print(f"  Format: AVI (MJPG - compatible with Windows Media Player)")
                print(f"  Resolution: {width}x{height}")
                print(f"  FPS: {fps}")
            else:
                print("✗ VideoWriter failed, will save frames as images instead")
                writer = None
        
        print("\nControls:")
        print("  'q' - Quit")
        print("  't' - Toggle trails")
        print("  's' - Take screenshot")
        print("\nProcessing...")
        
        frame_count = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("End of video or error reading frame")
                    break
                
                # Process frame
                annotated_frame = self.process_frame(frame, show_trails)
                
                # Save frame if recording
                if writer and writer.isOpened():
                    writer.write(annotated_frame)
                elif save_output:
                    # Fallback: save every 10th frame as image
                    if frame_count % 10 == 0:
                        frames_to_save.append(annotated_frame.copy())
                
                # Display frame
                cv2.imshow('Object Detection and Tracking', annotated_frame)
                
                frame_count += 1
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("\nStopping...")
                    break
                elif key == ord('t'):
                    show_trails = not show_trails
                    print(f"Trails: {'ON' if show_trails else 'OFF'}")
                elif key == ord('s'):
                    screenshot_path = f'screenshot_{frame_count}.jpg'
                    cv2.imwrite(screenshot_path, annotated_frame)
                    print(f"Screenshot saved: {screenshot_path}")
        
        finally:
            print(f"\nProcessed {frame_count} frames")
            cap.release()
            
            if writer and writer.isOpened():
                writer.release()
                
                # Verify file was created
                import os
                if os.path.exists(save_output) and os.path.getsize(save_output) > 0:
                    file_size = os.path.getsize(save_output) / (1024 * 1024)  # MB
                    print(f"✓ Video saved successfully!")
                    print(f"  Location: {os.path.abspath(save_output)}")
                    print(f"  Size: {file_size:.2f} MB")
                else:
                    print(f"✗ Warning: Video file not created or is empty")
            elif save_output and len(frames_to_save) > 0:
                # Fallback: save as individual images
                import os
                output_dir = save_output.replace('.avi', '').replace('.mp4', '') + '_frames'
                os.makedirs(output_dir, exist_ok=True)
                print(f"Video codec failed. Saving {len(frames_to_save)} frames as images...")
                for i, frame in enumerate(frames_to_save):
                    cv2.imwrite(f"{output_dir}/frame_{i:04d}.jpg", frame)
                print(f"✓ Frames saved to: {os.path.abspath(output_dir)}")
            
            cv2.destroyAllWindows()


def main():
    """Main function to run the tracker"""
    print("=" * 60)
    print("Real-time Object Detection and Tracking System")
    print("=" * 60)
    
    # Choose video source
    print("\nSelect video source:")
    print("  1. Webcam")
    print("  2. Video file")
    choice = input("Enter choice (1 or 2): ").strip()
    
    if choice == '1':
        source = 0
        print("\nUsing webcam...")
    else:
        source = input("Enter video file path: ").strip()
    
    # Choose to save output
    save_choice = input("\nSave output video? (y/n): ").strip().lower()
    save_output = None
    if save_choice == 'y':
        filename = input("Enter output filename (e.g., output.mp4): ").strip()
        # Save in the same directory as the script
        import os
        script_dir = os.path.dirname(os.path.abspath(__file__))
        save_output = os.path.join(script_dir, filename)
        print(f"Video will be saved to: {save_output}")
    
    # Initialize and run tracker
    tracker = ObjectDetectionTracker(
        model_name='yolov8n.pt',  # Use nano model for speed
        confidence_threshold=0.5
    )
    
    tracker.run_video(source=source, show_trails=True, save_output=save_output)


if __name__ == "__main__":
    main()