import os
import sys
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import text

MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "universal_sentence_encoder.tflite")
DEFAULT_TEXT = "MediaPipe simplifies on-device machine learning workflows."


def _options():
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    return text.TextEmbedderOptions(base_options=base_options, l2_normalize=True)


def run(input_text: str = DEFAULT_TEXT):
    with text.TextEmbedder.create_from_options(_options()) as embedder:
        result = embedder.embed(input_text)

        print(f"--- Text Embedder Results ---")
        print(f"Input: \"{input_text}\"")
        if result.embeddings:
            vec = result.embeddings[0].embedding
            print(f"  Embedding dimension: {len(vec)}")
            print(f"  First 5 vector values: {[round(float(v), 4) for v in vec[:5]]}")
        else:
            print("No embedding produced.")


COMPARE_PAIRS = [
    # similar meaning, different wording -> high similarity
    ("I'm feeling so good today.", "I'm having a great day!"),
    # same topic, different phrasing -> moderately high similarity
    ("The cat is sleeping on the sofa.", "A kitten naps on the couch."),
    # unrelated meaning -> low similarity
    ("I'm feeling so good today.", "the weather is terrible outside."),
]


def compare(pairs=COMPARE_PAIRS):
    """Embed sentence pairs and print their cosine similarity (MediaPipe demo use case)."""
    with text.TextEmbedder.create_from_options(_options()) as embedder:
        cache = {}

        def embed(sentence):
            if sentence not in cache:
                cache[sentence] = embedder.embed(sentence).embeddings[0]
            return cache[sentence]

        print("--- Text Embedder Similarity ---")
        for a, b in pairs:
            sim = text.TextEmbedder.cosine_similarity(embed(a), embed(b))
            print(f"  \"{a}\"\n    vs \"{b}\"\n    cosine similarity: {sim:.4f}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--compare":
        if len(sys.argv) == 4:
            compare([(sys.argv[2], sys.argv[3])])
        else:
            compare()
    else:
        t = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_TEXT
        run(t)
compare()