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


MODEL_PATH = os.path.join(os.path.dirname(__file__), "blaze_face_short_range.tflite")


def classify_face(image_path: str) -> str:
    """
    Detects face in the image and classifies looking direction:
    Returns: 'left', 'right', or 'straight'
    """
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.FaceDetectorOptions(base_options=base_options)
    detector = vision.FaceDetector.create_from_options(options)

    image = mp.Image.create_from_file(image_path)
    detection_result = detector.detect(image)

    if not detection_result.detections:
        return "straight"

    detection = detection_result.detections[0]
    keypoints = detection.keypoints

    # FaceDetector returns 6 keypoints:
    # 0: right eye, 1: left eye, 2: nose tip, 3: mouth center, 4: right ear, 5: left ear
    right_eye = keypoints[0]
    left_eye = keypoints[1]
    nose = keypoints[2]

    # Calculate horizontal offset of nose relative to the midpoint between eyes,
    # normalized by the inter-ocular distance (scale and resolution invariant).
    mid_eye_x = (right_eye.x + left_eye.x) / 2.0
    eye_distance = abs(left_eye.x - right_eye.x)
    if eye_distance == 0:
        return "straight"

    delta = (nose.x - mid_eye_x) / eye_distance

    # delta < -0.15: Nose is shifted left -> looking left
    # delta > 0.15: Nose is shifted right -> looking right
    # Otherwise: Centered -> looking straight
    if delta < -0.15:
        return "left"
    elif delta > 0.15:
        return "right"
    else:
        return "straight"


class Predictor(BasePredictor):
    """Cog container predictor."""
    def predict(
        self,
        image: Path = Input(description="Input image to classify face looking direction")
    ) -> str:
        return classify_face(str(image))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 classify_face.py <IMAGE_PATH>")
        sys.exit(1)

    result = classify_face(sys.argv[1])
    print(result)
