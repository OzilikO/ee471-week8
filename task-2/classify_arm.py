import sys
import os
import ctypes
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Load local OpenGL library if present (for WSL host compatibility)
_gles = os.path.expanduser("~/.local/lib/libGLESv2.so.2")
if os.path.exists(_gles):
    ctypes.CDLL(_gles, mode=ctypes.RTLD_GLOBAL)

# Optional Cog import so the script runs with or without Cog installed
try:
    from cog import BasePredictor, Input, Path
except ImportError:
    class BasePredictor:
        pass

    def Input(*args, **kwargs):
        return None

    from pathlib import Path


MODEL_PATH = os.path.join(os.path.dirname(__file__), "pose_landmarker.task")


def classify_arm(image_path: str) -> str:
    """
    Detects person's pose and classifies which arm is raised:
    Returns: 'left', 'right', 'both', or 'None'
    """
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.PoseLandmarkerOptions(base_options=base_options)
    detector = vision.PoseLandmarker.create_from_options(options)

    image = mp.Image.create_from_file(image_path)
    detection_result = detector.detect(image)

    if not detection_result.pose_landmarks:
        return "None"

    landmarks = detection_result.pose_landmarks[0]

    # Keypoint indices:
    # 11: left_shoulder, 15: left_wrist
    # 12: right_shoulder, 16: right_wrist
    left_shoulder_y = landmarks[11].y
    left_wrist_y = landmarks[15].y
    right_shoulder_y = landmarks[12].y
    right_wrist_y = landmarks[16].y

    # In image coordinates, y=0 is top, y=1 is bottom.
    # A smaller y means the wrist is higher than the shoulder.
    left_up = left_wrist_y < left_shoulder_y
    right_up = right_wrist_y < right_shoulder_y

    if left_up and right_up:
        return "both"
    elif left_up:
        return "left"
    elif right_up:
        return "right"
    else:
        return "None"


class Predictor(BasePredictor):
    """Cog container predictor."""
    def predict(
        self,
        image: Path = Input(description="Input image to classify which arm is raised")
    ) -> str:
        return classify_arm(str(image))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 classify_arm.py <IMAGE_PATH>")
        sys.exit(1)

    result = classify_arm(sys.argv[1])
    print(result)
