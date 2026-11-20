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

MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "face_landmarker.task")
DEFAULT_IMG = os.path.join(os.path.dirname(__file__), "sample_data", "sample_face.png")
OUTPUT_IMG = os.path.join(os.path.dirname(__file__), "sample_data", "annotated_face_landmarker.png")


def run(image_path: str = DEFAULT_IMG, output_path: str = OUTPUT_IMG):
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.FaceLandmarkerOptions(base_options=base_options)
    
    with vision.FaceLandmarker.create_from_options(options) as detector:
        image = mp.Image.create_from_file(image_path)
        result = detector.detect(image)
        
        print(f"--- Face Landmarker Results ({os.path.basename(image_path)}) ---")
        if result.face_landmarks:
            landmarks = result.face_landmarks[0]
            print(f"Face mesh detected with {len(landmarks)} landmarks.")
            print(f"  Landmark 0 (lips/nose): (x={landmarks[0].x:.3f}, y={landmarks[0].y:.3f})")
            
            # Draw facial mesh tesselation lines and contours
            annotated_image = cv2.imread(image_path)
            for face_landmarks in result.face_landmarks:
                drawing_utils.draw_landmarks(
                    image=annotated_image,
                    landmark_list=face_landmarks,
                    connections=vision.FaceLandmarksConnections.FACE_LANDMARKS_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=drawing_styles.get_default_face_mesh_tesselation_style()
                )
                drawing_utils.draw_landmarks(
                    image=annotated_image,
                    landmark_list=face_landmarks,
                    connections=vision.FaceLandmarksConnections.FACE_LANDMARKS_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=drawing_styles.get_default_face_mesh_contours_style()
                )
            cv2.imwrite(output_path, annotated_image)
            print(f"Annotated image saved to: {output_path}")
        else:
            print("No face landmarks detected.")


if __name__ == "__main__":
    img = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_IMG
    run(img)
