import os
import sys
import ctypes
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

_gles = os.path.expanduser("~/.local/lib/libGLESv2.so.2")
if os.path.exists(_gles):
    ctypes.CDLL(_gles, mode=ctypes.RTLD_GLOBAL)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "efficientnet_lite0.tflite")
DEFAULT_IMG = os.path.join(os.path.dirname(__file__), "sample_data", "burger.jpg")


def run(image_path: str = DEFAULT_IMG):
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.ImageClassifierOptions(base_options=base_options, max_results=3)
    
    with vision.ImageClassifier.create_from_options(options) as classifier:
        image = mp.Image.create_from_file(image_path)
        result = classifier.classify(image)
        
        print(f"--- Image Classifier Results ({os.path.basename(image_path)}) ---")
        if result.classifications and result.classifications[0].categories:
            for cat in result.classifications[0].categories:
                print(f"  Category: {cat.category_name} (Score: {cat.score:.3f})")
        else:
            print("No classifications returned.")


if __name__ == "__main__":
    img = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_IMG
    run(img)
