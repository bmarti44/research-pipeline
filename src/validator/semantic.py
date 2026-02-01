"""Semantic classification for query types."""

import json
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

import numpy as np
from sentence_transformers import SentenceTransformer

from .exemplars import STATIC_KNOWLEDGE_EXEMPLARS, MEMORY_REFERENCE_EXEMPLARS


@dataclass
class Thresholds:
    static_knowledge: float = 0.40
    memory_reference: float = 0.50
    duplicate_search: float = 0.85


class SemanticClassifier:
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        thresholds_path: Optional[Path] = None,
    ):
        self.model = SentenceTransformer(model_name)
        self.thresholds = self._load_thresholds(thresholds_path)

        self._static_centroid = self._compute_centroid(STATIC_KNOWLEDGE_EXEMPLARS)
        self._memory_centroid = self._compute_centroid(MEMORY_REFERENCE_EXEMPLARS)

    def _load_thresholds(self, path: Optional[Path]) -> Thresholds:
        if path is None:
            path = Path(__file__).parent.parent.parent / "calibration" / "thresholds.json"

        if path.exists():
            with open(path) as f:
                data = json.load(f)
            return Thresholds(
                static_knowledge=data.get("static_knowledge", {}).get("threshold", 0.60),
                memory_reference=data.get("memory_reference", {}).get("threshold", 0.55),
                duplicate_search=data.get("duplicate_search", {}).get("threshold", 0.85),
            )
        return Thresholds()

    def _compute_centroid(self, texts: list[str]) -> np.ndarray:
        embeddings = self.model.encode(texts)
        return np.mean(embeddings, axis=0)

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    def is_static_knowledge_query(self, query: str) -> tuple[bool, float]:
        query_embedding = self.model.encode(query)
        similarity = self._cosine_similarity(query_embedding, self._static_centroid)
        return similarity >= self.thresholds.static_knowledge, similarity

    def is_memory_reference_query(self, query: str) -> tuple[bool, float]:
        query_embedding = self.model.encode(query)
        similarity = self._cosine_similarity(query_embedding, self._memory_centroid)
        return similarity >= self.thresholds.memory_reference, similarity

    def is_duplicate_search(self, query: str, prior_queries: list[str]) -> tuple[bool, float]:
        if not prior_queries:
            return False, 0.0

        query_embedding = self.model.encode(query)
        prior_embeddings = self.model.encode(prior_queries)

        similarities = [self._cosine_similarity(query_embedding, pe) for pe in prior_embeddings]
        best_score = max(similarities)

        return best_score >= self.thresholds.duplicate_search, best_score
