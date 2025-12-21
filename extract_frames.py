import cv2
import os
from pathlib import Path
import sys

# Configuration
VIDEOS_DIR = "Turret/data/videos1"
OUTPUT_BASE_DIR = "Turret/data/frames"
FPS_TARGET = 30

def extract_frames_from_video(video_path, output_dir, target_fps=30):
    """Extract frames from a video at the specified FPS."""
    
    try:
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Open the video
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return False
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Calculate frame interval (every Nth frame to achieve target FPS)
        frame_interval = max(1, int(fps / target_fps))
        
        print(f"\nProcessing: {os.path.basename(video_path)}")
        print(f"  Original FPS: {fps}, Target FPS: {target_fps}")
        print(f"  Frame interval: {frame_interval}")
        print(f"  Total frames: {total_frames}")
        print(f"  Output: {output_dir}", flush=True)
        
        frame_count = 0
        saved_count = 0
        
        while True:
            ret, frame = cap.read()
            
            if not ret:
                break
            
            # Save frame at specified interval
            if frame_count % frame_interval == 0:
                frame_filename = os.path.join(output_dir, f"frame_{saved_count:06d}.jpg")
                cv2.imwrite(frame_filename, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                saved_count += 1
                
                if saved_count % 100 == 0:
                    print(f"  Saved {saved_count} frames...", flush=True)
            
            frame_count += 1
        
        cap.release()
        
        print(f"  ✓ Completed: {saved_count} frames saved")
        return True
        
    except Exception as e:
        print(f"  ✗ Error: {str(e)}")
        return False

def main():
    """Extract frames from all videos in the videos directory."""
    
    # Get all MP4 files
    video_files = sorted(Path(VIDEOS_DIR).glob("*.mp4"))
    
    if not video_files:
        print(f"No MP4 files found in {VIDEOS_DIR}")
        return
    
    print(f"Found {len(video_files)} video(s) to process\n")
    
    success_count = 0
    for i, video_file in enumerate(video_files, 1):
        # Create output directory based on video timestamp
        video_name = video_file.stem  # e.g., "camera_20250626_173636"
        timestamp = video_name.replace("camera_", "")
        output_dir = os.path.join(OUTPUT_BASE_DIR, timestamp)
        
        print(f"[{i}/{len(video_files)}]", end=" ")
        if extract_frames_from_video(str(video_file), output_dir, FPS_TARGET):
            success_count += 1
    
    print(f"\n{'='*50}")
    print(f"Extraction complete: {success_count}/{len(video_files)} successful")
    print(f"Frames saved to: {OUTPUT_BASE_DIR}/")
    sys.stdout.flush()

if __name__ == "__main__":
    main()
