import os
import sys
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python.retrieval import universal_embedder

MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "embeddinggemma-2-text-270m.litertlm")
VISION_MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "embeddinggemma-2-text-vision-440m.litertlm")
DEFAULT_TEXT = "MediaPipe multi-modal embeddings across vision, text, and audio."
DEFAULT_IMAGE = os.path.join(os.path.dirname(__file__), "sample_data", "burger.jpg")


def _options(model_path: str = MODEL_PATH):
    base_options = python.BaseOptions(model_asset_path=model_path)
    return universal_embedder.UniversalEmbedderOptions(base_options=base_options, l2_normalize=True)


def run(input_text: str = DEFAULT_TEXT):
    with universal_embedder.UniversalEmbedder.create_from_options(_options()) as embedder:
        result = embedder.embed_text(input_text)

        print(f"--- Universal Embedder Results ---")
        print(f"Input: \"{input_text}\"")
        if result and hasattr(result, "embeddings"):
            vec = result.embeddings[0].embedding
            print(f"  Multi-modal embedding dimension: {len(vec)}")
            print(f"  First 5 vector values: {[round(float(v), 4) for v in vec[:5]]}")
        else:
            print("Embedding extraction completed.")


COMPARE_PAIRS = [
    # similar meaning, different wording -> high similarity
    ("I'm feeling so good today.", "I'm having a great day!"),
    # same topic, different phrasing -> moderately high similarity
    ("The cat is sleeping on the sofa.", "A kitten naps on the couch."),
    # unrelated meaning -> low similarity
    ("I'm feeling so good today.", "the weather is terrible outside."),
]


def compare(pairs=COMPARE_PAIRS):
    """Embed text pairs and print their cosine similarity (MediaPipe demo use case)."""
    with universal_embedder.UniversalEmbedder.create_from_options(_options()) as embedder:
        cache = {}

        def embed(sentence):
            if sentence not in cache:
                cache[sentence] = embedder.embed_text(sentence).embeddings[0]
            return cache[sentence]

        print("--- Universal Embedder Similarity ---")
        for a, b in pairs:
            sim = universal_embedder.UniversalEmbedder.cosine_similarity(embed(a), embed(b))
            print(f"  \"{a}\"\n    vs \"{b}\"\n    cosine similarity: {sim:.4f}")


IMAGE_CAPTIONS = [
    # matching description -> highest similarity
    "A juicy cheeseburger with lettuce and tomato.",
    # related food, different dish -> medium similarity
    "A slice of pepperoni pizza.",
    # unrelated -> low similarity
    "A red sports car parked on the street.",
]


def compare_image(image_path: str = DEFAULT_IMAGE, captions=IMAGE_CAPTIONS):
    """Embed an image and several captions in a shared space, then rank captions by similarity.

    Requires the text+vision model; the text-only 270m model has no vision encoder.
    """
    with open(image_path, "rb") as f:
        image_bytes = f.read()

    with universal_embedder.UniversalEmbedder.create_from_options(_options(VISION_MODEL_PATH)) as embedder:
        image_emb = embedder.embed_image(image_bytes).embeddings[0]

        scores = []
        for caption in captions:
            text_emb = embedder.embed_text(caption).embeddings[0]
            scores.append((universal_embedder.UniversalEmbedder.cosine_similarity(image_emb, text_emb), caption))

        print("--- Universal Embedder Text vs Image ---")
        print(f"Image: {os.path.basename(image_path)}")
        for sim, caption in sorted(scores, reverse=True):
            print(f"  {sim:.4f}  \"{caption}\"")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--image":
        # --image [path] [caption ...]
        img = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_IMAGE
        caps = sys.argv[3:] or IMAGE_CAPTIONS
        compare_image(img, caps)
    elif len(sys.argv) > 1 and sys.argv[1] == "--compare":
        if len(sys.argv) == 4:
            compare([(sys.argv[2], sys.argv[3])])
        else:
            compare()
    else:
        t = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_TEXT
        run(t)
compare_image()