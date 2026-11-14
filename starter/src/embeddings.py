"""Local Sentence Transformers adapter for one consistent document/query model."""
from typing import Protocol


DEFAULT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


class Embedder(Protocol):
    model_name: str
    dimensions: int

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Return one vector per input text, in the same order."""
        ...

    def embed_query(self, question: str) -> list[float]:
        """Return a query vector compatible with the stored document vectors."""
        ...


class MyEmbedder:
    """Embed text with a real, normalized Sentence Transformers model."""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name.strip() or DEFAULT_MODEL
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "sentence-transformers is required; install starter/requirements.txt"
            ) from exc
        self._model = SentenceTransformer(self.model_name)
        dimensions = self._model.get_sentence_embedding_dimension()
        if not dimensions:
            raise RuntimeError(f"Embedding model has no dimension: {self.model_name}")
        self.dimensions = int(dimensions)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors = self._model.encode(
            texts, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False
        )
        return vectors.tolist()

    def embed_query(self, question: str) -> list[float]:
        if not question.strip():
            raise ValueError("question must not be empty")
        return self.embed_documents([question])[0]
