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

MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "holistic_landmarker.task")
DEFAULT_IMG = os.path.join(os.path.dirname(__file__), "sample_data", "sample_pose.jpg")
OUTPUT_IMG = os.path.join(os.path.dirname(__file__), "sample_data", "annotated_holistic_landmarker.jpg")


def run(image_path: str = DEFAULT_IMG, output_path: str = OUTPUT_IMG):
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.HolisticLandmarkerOptions(base_options=base_options)
    
    with vision.HolisticLandmarker.create_from_options(options) as detector:
        image = mp.Image.create_from_file(image_path)
        result = detector.detect(image)
        
        print(f"--- Holistic Landmarker Results ({os.path.basename(image_path)}) ---")
        pose_count = len(result.pose_landmarks) if result.pose_landmarks else 0
        face_count = len(result.face_landmarks) if result.face_landmarks else 0
        lh_count = len(result.left_hand_landmarks) if result.left_hand_landmarks else 0
        rh_count = len(result.right_hand_landmarks) if result.right_hand_landmarks else 0
        print(f"  Pose landmarks: {pose_count}")
        print(f"  Face landmarks: {face_count}")
        print(f"  Left hand landmarks: {lh_count}")
        print(f"  Right hand landmarks: {rh_count}")

        annotated_image = cv2.imread(image_path)
        # Draw pose
        if result.pose_landmarks:
            drawing_utils.draw_landmarks(
                annotated_image, result.pose_landmarks,
                vision.PoseLandmarksConnections.POSE_LANDMARKS,
                drawing_styles.get_default_pose_landmarks_style()
            )
        # Draw face mesh
        if result.face_landmarks:
            drawing_utils.draw_landmarks(
                annotated_image, result.face_landmarks,
                vision.FaceLandmarksConnections.FACE_LANDMARKS_TESSELATION,
                None,
                drawing_styles.get_default_face_mesh_tesselation_style()
            )
        # Draw hands
        if result.left_hand_landmarks:
            drawing_utils.draw_landmarks(
                annotated_image, result.left_hand_landmarks,
                vision.HandLandmarksConnections.HAND_CONNECTIONS,
                drawing_styles.get_default_hand_landmarks_style(),
                drawing_styles.get_default_hand_connections_style()
            )
        if result.right_hand_landmarks:
            drawing_utils.draw_landmarks(
                annotated_image, result.right_hand_landmarks,
                vision.HandLandmarksConnections.HAND_CONNECTIONS,
                drawing_styles.get_default_hand_landmarks_style(),
                drawing_styles.get_default_hand_connections_style()
            )
        
        cv2.imwrite(output_path, annotated_image)
        print(f"Annotated image saved to: {output_path}")


if __name__ == "__main__":
    img = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_IMG
    run(img)
