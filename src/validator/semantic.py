"""Semantic classification for query types."""

import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from sentence_transformers import SentenceTransformer

from .exemplars import (
    STATIC_KNOWLEDGE_EXEMPLARS,
    MEMORY_REFERENCE_EXEMPLARS,
    STATIC_KNOWLEDGE_HOLDOUT,
    MEMORY_REFERENCE_HOLDOUT,
)


@dataclass
class Thresholds:
    static_knowledge: float = 0.45  # Raised to reduce false positives on current-events queries
    memory_reference: float = 0.45  # Catch memory references
    duplicate_search: float = 0.80  # Catch near-duplicate searches
    duplicate_file_read: float = 0.85  # F16: Catch duplicate file reads
    cascading_search: float = 0.75  # F20: Catch narrowing searches
    answer_in_context: float = 0.70  # F19: Catch searching for info already provided


@dataclass
class CrossValidationResult:
    """Result of k-fold cross-validation for threshold selection."""
    optimal_threshold: float
    mean_accuracy: float
    std_accuracy: float
    fold_results: list[dict] = field(default_factory=list)
    auc: Optional[float] = None  # Area under ROC curve


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
                duplicate_file_read=data.get("duplicate_file_read", {}).get("threshold", 0.85),
                cascading_search=data.get("cascading_search", {}).get("threshold", 0.75),
                answer_in_context=data.get("answer_in_context", {}).get("threshold", 0.70),
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

    def is_duplicate_file_read(self, file_path: str, prior_reads: list[str]) -> tuple[bool, float]:
        """F16: Check if a file read is a duplicate of a prior read.

        Uses semantic similarity to catch variations like:
        - Same file with different relative/absolute paths
        - Slightly different path formats
        """
        if not prior_reads:
            return False, 0.0

        # Normalize paths for comparison
        path_embedding = self.model.encode(file_path)
        prior_embeddings = self.model.encode(prior_reads)

        similarities = [self._cosine_similarity(path_embedding, pe) for pe in prior_embeddings]
        best_score = max(similarities)

        return best_score >= self.thresholds.duplicate_file_read, best_score

    def is_cascading_search(self, query: str, prior_queries: list[str]) -> tuple[bool, float]:
        """F20: Check if search is a narrowing of a prior search.

        Detects patterns like:
        - "Python" -> "Python tutorial" -> "Python beginner tutorial"

        Returns True if the new query semantically contains a prior query
        (is more specific, making the prior search wasteful).
        """
        if not prior_queries:
            return False, 0.0

        query_embedding = self.model.encode(query)
        prior_embeddings = self.model.encode(prior_queries)

        # Check if new query is highly similar but longer (more specific)
        for i, prior in enumerate(prior_queries):
            similarity = self._cosine_similarity(query_embedding, prior_embeddings[i])

            # Cascading pattern: high similarity + new query is more specific (longer)
            if similarity >= self.thresholds.cascading_search:
                # Check if this is a narrowing (query contains or extends prior)
                query_lower = query.lower()
                prior_lower = prior.lower()

                # If prior terms are subset of query terms, it's cascading
                prior_words = set(prior_lower.split())
                query_words = set(query_lower.split())

                if prior_words.issubset(query_words) and len(query_words) > len(prior_words):
                    return True, similarity

                # Also catch semantic narrowing even without exact word overlap
                if len(query) > len(prior) * 1.2 and similarity >= self.thresholds.cascading_search:
                    return True, similarity

        return False, 0.0

    def is_answer_in_context(self, search_query: str, user_context: str) -> tuple[bool, float]:
        """F19: Check if the search query is looking for info already in context.

        Detects when user already provided the information being searched for.
        """
        if not user_context:
            return False, 0.0

        # Encode both
        query_embedding = self.model.encode(search_query)
        context_embedding = self.model.encode(user_context)

        similarity = self._cosine_similarity(query_embedding, context_embedding)

        return similarity >= self.thresholds.answer_in_context, similarity

    def cross_validate_threshold(
        self,
        positive_examples: list[str],
        negative_examples: list[str],
        category: str,
        k_folds: int = 5,
        threshold_range: tuple[float, float] = (0.30, 0.70),
        threshold_steps: int = 20,
    ) -> CrossValidationResult:
        """
        Perform k-fold cross-validation to find optimal threshold.

        Args:
            positive_examples: Examples that SHOULD match (e.g., static knowledge queries)
            negative_examples: Examples that should NOT match (e.g., current events queries)
            category: 'static_knowledge' or 'memory_reference'
            k_folds: Number of folds for cross-validation
            threshold_range: (min, max) threshold values to test
            threshold_steps: Number of threshold values to try

        Returns:
            CrossValidationResult with optimal threshold and metrics
        """
        # Encode all examples
        pos_embeddings = self.model.encode(positive_examples)
        neg_embeddings = self.model.encode(negative_examples)

        # Generate threshold candidates
        thresholds = np.linspace(threshold_range[0], threshold_range[1], threshold_steps)

        fold_results = []
        all_labels = []
        all_scores = []

        # Create folds
        n_pos = len(positive_examples)
        n_neg = len(negative_examples)
        pos_indices = np.arange(n_pos)
        neg_indices = np.arange(n_neg)

        np.random.shuffle(pos_indices)
        np.random.shuffle(neg_indices)

        pos_folds = np.array_split(pos_indices, k_folds)
        neg_folds = np.array_split(neg_indices, k_folds)

        for fold in range(k_folds):
            # Held-out test set
            test_pos_idx = pos_folds[fold]
            test_neg_idx = neg_folds[fold]

            # Training set (all other folds)
            train_pos_idx = np.concatenate([pos_folds[i] for i in range(k_folds) if i != fold])
            train_neg_idx = np.concatenate([neg_folds[i] for i in range(k_folds) if i != fold])

            # Compute centroid from training positive examples only
            train_centroid = np.mean(pos_embeddings[train_pos_idx], axis=0)

            # Compute similarities for test examples
            test_pos_sims = [self._cosine_similarity(pos_embeddings[i], train_centroid)
                            for i in test_pos_idx]
            test_neg_sims = [self._cosine_similarity(neg_embeddings[i], train_centroid)
                            for i in test_neg_idx]

            # Store for ROC curve
            all_scores.extend(test_pos_sims + test_neg_sims)
            all_labels.extend([1] * len(test_pos_sims) + [0] * len(test_neg_sims))

            # Find best threshold for this fold
            best_threshold = thresholds[0]
            best_accuracy = 0.0

            for thresh in thresholds:
                # Predictions
                pos_correct = sum(1 for s in test_pos_sims if s >= thresh)
                neg_correct = sum(1 for s in test_neg_sims if s < thresh)
                accuracy = (pos_correct + neg_correct) / (len(test_pos_sims) + len(test_neg_sims))

                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_threshold = thresh

            fold_results.append({
                "fold": fold,
                "best_threshold": float(best_threshold),
                "accuracy": float(best_accuracy),
                "n_test_pos": len(test_pos_idx),
                "n_test_neg": len(test_neg_idx),
            })

        # Compute AUC using trapezoidal rule
        sorted_pairs = sorted(zip(all_scores, all_labels), reverse=True)
        n_pos_total = sum(all_labels)
        n_neg_total = len(all_labels) - n_pos_total

        tpr_prev, fpr_prev = 0.0, 0.0
        auc = 0.0
        tp, fp = 0, 0

        for score, label in sorted_pairs:
            if label == 1:
                tp += 1
            else:
                fp += 1
            tpr = tp / n_pos_total if n_pos_total > 0 else 0
            fpr = fp / n_neg_total if n_neg_total > 0 else 0
            auc += (fpr - fpr_prev) * (tpr + tpr_prev) / 2
            tpr_prev, fpr_prev = tpr, fpr

        # Optimal threshold is mean of fold thresholds
        optimal_threshold = float(np.mean([f["best_threshold"] for f in fold_results]))
        mean_accuracy = float(np.mean([f["accuracy"] for f in fold_results]))
        std_accuracy = float(np.std([f["accuracy"] for f in fold_results]))

        return CrossValidationResult(
            optimal_threshold=optimal_threshold,
            mean_accuracy=mean_accuracy,
            std_accuracy=std_accuracy,
            fold_results=fold_results,
            auc=float(auc),
        )

    def validate_on_holdout(self, category: str = "static_knowledge") -> dict:
        """
        Validate classifier performance on held-out data to detect overfitting.

        This method uses STATIC_KNOWLEDGE_HOLDOUT or MEMORY_REFERENCE_HOLDOUT
        (which are NOT used in training) to test generalization.

        Args:
            category: "static_knowledge" or "memory_reference"

        Returns:
            Dictionary with accuracy and per-example results
        """
        if category == "static_knowledge":
            holdout = STATIC_KNOWLEDGE_HOLDOUT
            check_func = self.is_static_knowledge_query
        elif category == "memory_reference":
            holdout = MEMORY_REFERENCE_HOLDOUT
            check_func = self.is_memory_reference_query
        else:
            raise ValueError(f"Unknown category: {category}")

        results = []
        correct = 0
        for query in holdout:
            is_match, score = check_func(query)
            results.append({
                "query": query,
                "predicted": is_match,
                "expected": True,  # Holdout examples should match
                "score": score,
                "correct": is_match,  # Should be True for positive holdout examples
            })
            if is_match:
                correct += 1

        return {
            "category": category,
            "accuracy": correct / len(holdout) if holdout else 0.0,
            "n_total": len(holdout),
            "n_correct": correct,
            "results": results,
        }
