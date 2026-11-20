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

MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "efficientdet_lite0.tflite")
DEFAULT_IMG = os.path.join(os.path.dirname(__file__), "sample_data", "cat.jpg")
OUTPUT_IMG = os.path.join(os.path.dirname(__file__), "sample_data", "annotated_object_detector.jpg")


def run(image_path: str = DEFAULT_IMG, output_path: str = OUTPUT_IMG):
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.ObjectDetectorOptions(
        base_options=base_options,
        score_threshold=0.3
    )
    
    with vision.ObjectDetector.create_from_options(options) as detector:
        image = mp.Image.create_from_file(image_path)
        result = detector.detect(image)
        
        print(f"--- Object Detector Results ({os.path.basename(image_path)}) ---")
        print(f"Objects detected: {len(result.detections)}")
        
        annotated_image = cv2.imread(image_path)
        for idx, detection in enumerate(result.detections):
            category = detection.categories[0]
            bbox = detection.bounding_box
            print(f"  Object #{idx+1}: {category.category_name} (Score={category.score:.2f}) at Box=[x={bbox.origin_x}, y={bbox.origin_y}, w={bbox.width}, h={bbox.height}]")
            drawing_utils.draw_detection(annotated_image, detection)
            
            # Put label text above box
            cv2.putText(
                annotated_image,
                f"{category.category_name} {category.score:.2f}",
                (bbox.origin_x, max(0, bbox.origin_y - 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
            )
            
        cv2.imwrite(output_path, annotated_image)
        print(f"Annotated image saved to: {output_path}")


if __name__ == "__main__":
    img = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_IMG
    run(img)
