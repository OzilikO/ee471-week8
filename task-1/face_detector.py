import os
import sys
import ctypes
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import drawing_utils

_gles = os.path.expanduser("~/.local/lib/libGLESv2.so.2")
if os.path.exists(_gles):
    ctypes.CDLL(_gles, mode=ctypes.RTLD_GLOBAL)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "blaze_face_short_range.tflite")
DEFAULT_IMG = os.path.join(os.path.dirname(__file__), "sample_data", "sample_face.png")
OUTPUT_IMG = os.path.join(os.path.dirname(__file__), "sample_data", "annotated_face_detector.png")


def run(image_path: str = DEFAULT_IMG, output_path: str = OUTPUT_IMG):
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.FaceDetectorOptions(base_options=base_options)
    
    with vision.FaceDetector.create_from_options(options) as detector:
        image = mp.Image.create_from_file(image_path)
        result = detector.detect(image)
        
        print(f"--- Face Detector Results ({os.path.basename(image_path)}) ---")
        print(f"Faces detected: {len(result.detections)}")
        
        annotated_image = cv2.imread(image_path)
        for idx, detection in enumerate(result.detections):
            bbox = detection.bounding_box
            score = detection.categories[0].score if detection.categories else 0.0
            print(f"  Face #{idx+1}: Box=[x={bbox.origin_x}, y={bbox.origin_y}, w={bbox.width}, h={bbox.height}], Score={score:.2f}")
            drawing_utils.draw_detection(annotated_image, detection)
        
        cv2.imwrite(output_path, annotated_image)
        print(f"Annotated image saved to: {output_path}")


if __name__ == "__main__":
    img = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_IMG
    run(img)
