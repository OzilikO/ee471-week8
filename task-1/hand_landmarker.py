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

MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "hand_landmarker.task")
DEFAULT_IMG = os.path.join(os.path.dirname(__file__), "sample_data", "sample_gesture.jpg")
OUTPUT_IMG = os.path.join(os.path.dirname(__file__), "sample_data", "annotated_hand_landmarker.jpg")


def run(image_path: str = DEFAULT_IMG, output_path: str = OUTPUT_IMG):
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.HandLandmarkerOptions(base_options=base_options)
    
    with vision.HandLandmarker.create_from_options(options) as detector:
        image = mp.Image.create_from_file(image_path)
        result = detector.detect(image)
        
        print(f"--- Hand Landmarker Results ({os.path.basename(image_path)}) ---")
        annotated_image = cv2.imread(image_path)
        if result.hand_landmarks:
            for hand_idx, landmarks in enumerate(result.hand_landmarks):
                handedness = result.handedness[hand_idx][0].category_name if result.handedness else "Hand"
                print(f"  {handedness} detected ({len(landmarks)} landmarks). Wrist: (x={landmarks[0].x:.3f}, y={landmarks[0].y:.3f})")
                drawing_utils.draw_landmarks(
                    image=annotated_image,
                    landmark_list=landmarks,
                    connections=vision.HandLandmarksConnections.HAND_CONNECTIONS,
                    landmark_drawing_spec=drawing_styles.get_default_hand_landmarks_style(),
                    connection_drawing_spec=drawing_styles.get_default_hand_connections_style()
                )
            cv2.imwrite(output_path, annotated_image)
            print(f"Annotated image saved to: {output_path}")
        else:
            print("No hands detected.")


if __name__ == "__main__":
    img = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_IMG
    run(img)
