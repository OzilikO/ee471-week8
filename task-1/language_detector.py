import os
import sys
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import text

MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "language_detector.tflite")
DEFAULT_TEXT = "Bonjour tout le monde! C'est un exercice magnifique."


def run(input_text: str = DEFAULT_TEXT):
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = text.LanguageDetectorOptions(base_options=base_options)
    
    with text.LanguageDetector.create_from_options(options) as detector:
        result = detector.detect(input_text)
        
        print(f"--- Language Detector Results ---")
        print(f"Input: \"{input_text}\"")
        for detection in result.detections:
            print(f"  Detected language code: '{detection.language_code}' (Score: {detection.probability:.3f})")


if __name__ == "__main__":
    t = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_TEXT
    run(t)
