import os
import sys

def get_latest_session(base_dir="data/recordings"):
    """
    Find the most recent recording session folder.
    Also handles legacy flat-file recordings.
    """
    if not os.path.exists(base_dir):
        print("No recordings directory found!")
        sys.exit(1)
    
    # Check for timestamped subdirectories first (new format)
    subdirs = []
    for item in os.listdir(base_dir):
        item_path = os.path.join(base_dir, item)
        if os.path.isdir(item_path):
            # Check if it contains cam0.mp4 and cam1.mp4
            has_cam0 = os.path.exists(os.path.join(item_path, "cam0.mp4"))
            has_cam1 = os.path.exists(os.path.join(item_path, "cam1.mp4"))
            if has_cam0 and has_cam1:
                subdirs.append(item)
    
    if subdirs:
        latest = sorted(subdirs)[-1]
        session_path = os.path.join(base_dir, latest)
        print(f"Found session: {latest}")
        return session_path
    
    # Fallback: check for legacy flat files (old format: timestamp_cam0.mp4)
    cam0_files = sorted([f for f in os.listdir(base_dir) 
                         if f.endswith("_cam0.mp4")])
    if cam0_files:
        # Use the latest one
        latest_cam0 = cam0_files[-1]
        latest_cam1 = latest_cam0.replace("_cam0.mp4", "_cam1.mp4")
        
        if os.path.exists(os.path.join(base_dir, latest_cam1)):
            print(f"Found legacy recording: {latest_cam0}")
            print("Tip: New recordings save to subfolders automatically.")
            return base_dir  # Return base dir for legacy format
    
    print("No valid recordings found!")
    print("Run record_dual.py first to capture a session.")
    sys.exit(1)


def get_video_paths(session_dir):
    """
    Get cam0 and cam1 video paths from a session directory.
    Handles both new subfolder format and legacy flat format.
    """
    # New format: session_dir/cam0.mp4
    cam0 = os.path.join(session_dir, "cam0.mp4")
    cam1 = os.path.join(session_dir, "cam1.mp4")
    
    if os.path.exists(cam0) and os.path.exists(cam1):
        return cam0, cam1
    
    # Legacy format: session_dir/timestamp_cam0.mp4
    files = os.listdir(session_dir)
    cam0_files = sorted([f for f in files if f.endswith("_cam0.mp4")])
    
    if cam0_files:
        latest = cam0_files[-1]
        cam0 = os.path.join(session_dir, latest)
        cam1 = os.path.join(session_dir, latest.replace("_cam0.mp4", "_cam1.mp4"))
        if os.path.exists(cam1):
            return cam0, cam1
    
    print(f"Could not find cam0/cam1 videos in {session_dir}")
    sys.exit(1)


def get_processed_dir(session_dir):
    """Get or create the processed output directory for a session."""
    processed = os.path.join(session_dir, "processed")
    os.makedirs(processed, exist_ok=True)
    return processed