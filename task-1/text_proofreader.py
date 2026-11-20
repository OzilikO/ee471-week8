import os
import sys
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import text

MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "proofread_quant_200m.litertlm")
DEFAULT_TEXT = "She go to school everyday and she are very happy."


def run(input_text: str = DEFAULT_TEXT):
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file '{os.path.basename(MODEL_PATH)}' not found in models/.")
        print("Please ensure the model download has finished.")
        return

    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = text.TextProofreaderOptions(base_options=base_options)
    
    with text.TextProofreader.create_from_options(options) as proofreader:
        result = proofreader.proofread(input_text)
        
        print(f"--- Text Proofreader Results ---")
        print(f"Original Text: \"{input_text}\"")
        print(f"Proofread Text: \"{result.proofread_text}\"")

        changes = [c for c in result.corrections
                   if c.type != text.TextProofreaderCorrectionType.SAME]
        if changes:
            for idx, corr in enumerate(changes):
                print(f"  Change #{idx+1}: {corr.type.name:<9} '{corr.text}'")
        else:
            print("No corrections needed.")


if __name__ == "__main__":
    t = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_TEXT
    run(t)
