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

MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "pose_landmarker.task")
DEFAULT_IMG = os.path.join(os.path.dirname(__file__), "sample_data", "sample_pose.jpg")
OUTPUT_IMG = os.path.join(os.path.dirname(__file__), "sample_data", "annotated_pose_landmarker.jpg")


def run(image_path: str = DEFAULT_IMG, output_path: str = OUTPUT_IMG):
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.PoseLandmarkerOptions(base_options=base_options)
    
    with vision.PoseLandmarker.create_from_options(options) as detector:
        image = mp.Image.create_from_file(image_path)
        result = detector.detect(image)
        
        print(f"--- Pose Landmarker Results ({os.path.basename(image_path)}) ---")
        if result.pose_landmarks:
            landmarks = result.pose_landmarks[0]
            print(f"Pose detected with {len(landmarks)} landmarks.")
            print(f"  Nose:    (x={landmarks[0].x:.3f}, y={landmarks[0].y:.3f})")
            print(f"  L-Wrist: (x={landmarks[15].x:.3f}, y={landmarks[15].y:.3f})")
            print(f"  R-Wrist: (x={landmarks[16].x:.3f}, y={landmarks[16].y:.3f})")
            
            # Draw landmarks & skeleton connection lines
            annotated_image = cv2.imread(image_path)
            for pose_landmarks in result.pose_landmarks:
                drawing_utils.draw_landmarks(
                    image=annotated_image,
                    landmark_list=pose_landmarks,
                    connections=vision.PoseLandmarksConnections.POSE_LANDMARKS,
                    landmark_drawing_spec=drawing_styles.get_default_pose_landmarks_style()
                )
            cv2.imwrite(output_path, annotated_image)
            print(f"Annotated image saved to: {output_path}")
        else:
            print("No pose detected.")


if __name__ == "__main__":
    img = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_IMG
    run(img)
