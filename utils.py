import numpy as np

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle > 180.0:
        angle = 360-angle
        
    return angle

def is_finger_up(tip, mcp):
    # Check if finger is up based on y-coordinate
    return tip[1] < mcp[1]

def is_finger_down(tip, mcp):
    # Check if finger is down based on y-coordinate
    return tip[1] > mcp[1]

def classify_static_gesture(landmarks):
    # Finger tips
    thumb_tip = landmarks[4]
    index_tip = landmarks[8]
    middle_tip = landmarks[12]
    ring_tip = landmarks[16]
    pinky_tip = landmarks[20]

    # Finger MCP joints
    thumb_mcp = landmarks[2]
    index_mcp = landmarks[5]
    middle_mcp = landmarks[9]
    ring_mcp = landmarks[13]
    pinky_mcp = landmarks[17]

    # Check if fingers are up (tip y-coordinate < mcp y-coordinate)
    thumb_up = thumb_tip[1] < thumb_mcp[1] - 0.08
    index_up = index_tip[1] < index_mcp[1] - 0.08
    middle_up = middle_tip[1] < middle_mcp[1] - 0.08
    ring_up = ring_tip[1] < ring_mcp[1] - 0.08
    pinky_up = pinky_tip[1] < pinky_mcp[1] - 0.08

    # Count number of fingers up
    fingers_up = sum([thumb_up, index_up, middle_up, ring_up, pinky_up])

    # Calculate distances between fingers
    thumb_index_dist = np.linalg.norm(np.array(thumb_tip[:2]) - np.array(index_tip[:2]))
    index_middle_dist = np.linalg.norm(np.array(index_tip[:2]) - np.array(middle_tip[:2]))

    # Basic gesture recognition
    if fingers_up == 5:
        return "Stop"
    elif fingers_up == 4:
        return "Four"
    elif fingers_up == 3:
        return "Three"
    elif fingers_up == 2:
        if index_up and middle_up:
            return "Peace"
        elif thumb_up and index_up:
            return "Gun"
        elif thumb_up and pinky_up:
            return "Rock"
        elif index_up and pinky_up:
            return "Horns"
    elif fingers_up == 1:
        if thumb_up:
            return "Thumbs Up"
        elif index_up:
            return "Point"
        elif middle_up:
            return "Middle"
        elif ring_up:
            return "Ring"
        elif pinky_up:
            return "Pinky Up"
    elif fingers_up == 0:
        return "Fist"
    elif thumb_index_dist < 0.1:
        return "OK"

    return "Unknown"
