import os
import sys
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import text

MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "bert_classifier.tflite")
DEFAULT_TEXT = "I am having an absolutely wonderful day learning computer vision!"


def run(input_text: str = DEFAULT_TEXT):
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = text.TextClassifierOptions(base_options=base_options)
    
    with text.TextClassifier.create_from_options(options) as classifier:
        result = classifier.classify(input_text)
        
        print(f"--- Text Classifier Results ---")
        print(f"Input: \"{input_text}\"")
        if result.classifications:
            for cat in result.classifications[0].categories:
                print(f"  Category: '{cat.category_name}' (Score: {cat.score:.3f})")


if __name__ == "__main__":
    t = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_TEXT
    run(t)
