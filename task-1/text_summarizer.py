import os
import sys
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import text

MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "summarization_quant_200m_2modes.litertlm")
DEFAULT_TEXT = (
    "MediaPipe Solutions provides a suite of libraries and tools for you to quickly apply "
    "artificial intelligence (AI) and machine learning (ML) techniques in your applications. "
    "You can use these solutions across mobile platforms, web browsers, and desktop environments."
    "They help with several ML learning pipelines, including data collection, model training, and inference. "
)


def run(input_text: str = DEFAULT_TEXT):
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = text.TextSummarizerOptions(
        base_options=base_options,
        mode=text.TextSummarizerMode.TLDR,  # or text.TextSummarizerMode.TLDR
    )
    
    with text.TextSummarizer.create_from_options(options) as summarizer:
        result = summarizer.summarize(input_text)
        
        print(f"--- Text Summarizer Results ({options.mode.name}) ---")
        print(f"Input: \"{input_text}\"")
        print(f"Summary: \"{result.summary}\"")


if __name__ == "__main__":
    t = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_TEXT
    run(t)
