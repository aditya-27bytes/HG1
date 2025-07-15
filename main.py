import cv2
import mediapipe as mp
from gesture_recognition import GestureRecognizer

def main():
    cap = cv2.VideoCapture(0)
    
    # Default camera settings
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_BRIGHTNESS, 128)
    cap.set(cv2.CAP_PROP_CONTRAST, 128)
    cap.set(cv2.CAP_PROP_EXPOSURE, -1)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    recognizer = GestureRecognizer()
    gesture_history = []  # Gesture history for stability
    gesture_threshold = 5  # Number of frames to confirm gesture

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Frame processing
        frame = cv2.flip(frame, 1)
        
        # Hand detection and gesture recognition
        frame, current_gesture = recognizer.detect_and_classify(frame)
        
        # Gesture stability check
        gesture_history.append(current_gesture)
        if len(gesture_history) > gesture_threshold:
            gesture_history.pop(0)
        
        # Most common gesture in history
        if len(gesture_history) == gesture_threshold:
            final_gesture = max(set(gesture_history), key=gesture_history.count)
        else:
            final_gesture = current_gesture

        # Display gesture with better visibility
        cv2.putText(frame, f'Gesture: {final_gesture}', (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow('Hand Gesture Recognition', frame)

        if cv2.waitKey(10) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
