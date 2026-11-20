import os
import sys
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python.retrieval import semantic_retriever, universal_embedder

MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "embeddinggemma-2-text-270m.litertlm")
QUERY = "machine learning vision on mobile devices"
DOCUMENTS = [
    "MediaPipe offers on-device perception solutions for mobile and web developers.",
    "Baking sourdough bread requires flour, water, salt, and active wild yeast starter.",
    "Convolutional neural networks are commonly used for computer vision and object recognition."
]

def run(query: str = QUERY):
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    embedder_options = universal_embedder.UniversalEmbedderOptions(base_options=base_options)

    with universal_embedder.UniversalEmbedder.create_from_options(embedder_options) as embedder:
        options = semantic_retriever.SemanticRetrieverOptions(
            embedder=embedder,
            embedding_dimension=768,  # embeddinggemma output size
        )
        with semantic_retriever.SemanticRetriever.create_from_options(options) as retriever:
            for idx, doc in enumerate(DOCUMENTS):
                retriever.insert_document(doc_id=f"doc_{idx+1}", text=doc)

            print(f"--- Semantic Retriever Results ---")
            print(f"Query: \"{query}\"")
            results = retriever.retrieve(query, limit=2, min_similarity=0.0)
            for r in results:
                print(f"  Matched Doc: ID={r.id}, Score={r.score:.3f}")


if __name__ == "__main__":
    q = sys.argv[1] if len(sys.argv) > 1 else QUERY
    run(q)
