import os
import sys
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python.decision import decision_maker

MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "embeddinggemma-2-text-270m.litertlm")
QUERY = "A pedestrian is crossing the crosswalk ahead of the autonomous vehicle."


def run(query: str = QUERY):
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file '{os.path.basename(MODEL_PATH)}' not found in models/.")
        print("Please ensure the model download has finished.")
        return

    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = decision_maker.DecisionMakerOptions(base_options=base_options)
    
    with decision_maker.DecisionMaker.create_from_options(options) as decider:
        print(f"--- Decision Maker Results ---")
        print(f"Context: \"{query}\"")
        
        question = decision_maker.ChoiceQuestion(
            criteria={
                "yield": "Stop and yield to the pedestrian in the crosswalk",
                "proceed": "Keep accelerating forward without stopping"
            }
        )
        result = decider.evaluate_choice(query, question)
        print(f"Selected Decision: {result.selected_key}")
        print("Probabilities:")
        for k, v in result.probabilities.items():
            print(f"  {k}: {v:.3f}")


if __name__ == "__main__":
    q = sys.argv[1] if len(sys.argv) > 1 else QUERY
    run(q)
