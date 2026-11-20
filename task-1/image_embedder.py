import os
import sys
import ctypes
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

_gles = os.path.expanduser("~/.local/lib/libGLESv2.so.2")
if os.path.exists(_gles):
    ctypes.CDLL(_gles, mode=ctypes.RTLD_GLOBAL)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "mobilenet_v3_small.tflite")
DEFAULT_IMG = os.path.join(os.path.dirname(__file__), "sample_data", "burger.jpg")


def run(image_path: str = DEFAULT_IMG):
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.ImageEmbedderOptions(base_options=base_options, l2_normalize=True)
    
    with vision.ImageEmbedder.create_from_options(options) as embedder:
        image = mp.Image.create_from_file(image_path)
        result = embedder.embed(image)
        
        print(f"--- Image Embedder Results ({os.path.basename(image_path)}) ---")
        if result.embeddings:
            vec = result.embeddings[0].embedding
            print(f"  Embedding dimension: {len(vec)}")
            print(f"  First 5 vector values: {[round(float(v), 4) for v in vec[:5]]}")
        else:
            print("No embedding produced.")


SAMPLE_DIR = os.path.join(os.path.dirname(__file__), "sample_data")
COMPARE_PAIRS = [
    ("burger.jpg", "burger_crop.jpg"),  # same subject -> high similarity
    ("burger.jpg", "cat.jpg"),          # different subject -> low similarity
]


def compare(pairs=COMPARE_PAIRS):
    """Embed image pairs and print their cosine similarity (MediaPipe demo use case)."""
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.ImageEmbedderOptions(base_options=base_options, l2_normalize=True)

    with vision.ImageEmbedder.create_from_options(options) as embedder:
        cache = {}

        def embed(name):
            if name not in cache:
                image = mp.Image.create_from_file(os.path.join(SAMPLE_DIR, name))
                cache[name] = embedder.embed(image).embeddings[0]
            return cache[name]

        print("--- Image Embedder Similarity ---")
        for a, b in pairs:
            sim = vision.ImageEmbedder.cosine_similarity(embed(a), embed(b))
            print(f"  {a:<12} vs {b:<16} cosine similarity: {sim:.4f}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--compare":
        compare()
    else:
        img = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_IMG
        run(img)
compare(pairs=COMPARE_PAIRS)