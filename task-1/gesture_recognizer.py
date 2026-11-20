import os
import sys
import ctypes
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import drawing_utils, drawing_styles

_gles = os.path.expanduser("~/.local/lib/libGLESv2.so.2")
if os.path.exists(_gles):
    ctypes.CDLL(_gles, mode=ctypes.RTLD_GLOBAL)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "gesture_recognizer.task")
DEFAULT_IMG = os.path.join(os.path.dirname(__file__), "sample_data", "sample_gesture.jpg")
OUTPUT_IMG = os.path.join(os.path.dirname(__file__), "sample_data", "annotated_gesture_recognizer.jpg")


def run(image_path: str = DEFAULT_IMG, output_path: str = OUTPUT_IMG):
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.GestureRecognizerOptions(base_options=base_options)
    
    with vision.GestureRecognizer.create_from_options(options) as recognizer:
        image = mp.Image.create_from_file(image_path)
        result = recognizer.recognize(image)
        
        print(f"--- Gesture Recognizer Results ({os.path.basename(image_path)}) ---")
        annotated_image = cv2.imread(image_path)
        if result.gestures and result.hand_landmarks:
            for hand_idx, gesture_list in enumerate(result.gestures):
                top_gesture = gesture_list[0]
                handedness = result.handedness[hand_idx][0].category_name if result.handedness else "Hand"
                print(f"  {handedness}: Gesture='{top_gesture.category_name}' (Score={top_gesture.score:.2f})")
                
                landmarks = result.hand_landmarks[hand_idx]
                drawing_utils.draw_landmarks(
                    image=annotated_image,
                    landmark_list=landmarks,
                    connections=vision.HandLandmarksConnections.HAND_CONNECTIONS,
                    landmark_drawing_spec=drawing_styles.get_default_hand_landmarks_style(),
                    connection_drawing_spec=drawing_styles.get_default_hand_connections_style()
                )
                # Label gesture on image
                h, w, _ = annotated_image.shape
                cv2.putText(
                    annotated_image,
                    f"{top_gesture.category_name} ({top_gesture.score:.2f})",
                    (int(landmarks[0].x * w), int(landmarks[0].y * h) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
                )
            cv2.imwrite(output_path, annotated_image)
            print(f"Annotated image saved to: {output_path}")
        else:
            print("No gestures recognized.")


if __name__ == "__main__":
    img = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_IMG
    run(img)
