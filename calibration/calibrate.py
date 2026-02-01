"""
Threshold calibration for semantic classifier using ROC analysis.

This script:
1. Uses held-out calibration data (separate from test scenarios)
2. Performs k-fold cross-validation to find optimal thresholds
3. Computes ROC curves and AUC for each category
4. Saves calibrated thresholds to thresholds.json

Run: uv run python -m calibration.calibrate
"""

import json
from pathlib import Path
from dataclasses import asdict

import numpy as np

from src.validator.semantic import SemanticClassifier


# Calibration data: positive examples (should match) and negative examples (should not match)
# These are SEPARATE from both training exemplars and test scenarios

STATIC_KNOWLEDGE_CALIBRATION = {
    "positive": [
        # Factual questions about established knowledge
        "What is the atomic number of carbon?",
        "How does a transistor work?",
        "What is the Fibonacci sequence?",
        "Explain polymorphism in programming",
        "What causes tides?",
        "Who invented the telephone?",
        "What is the Heisenberg uncertainty principle?",
        "How does a compiler work?",
        "What is the difference between RAM and ROM?",
        "Explain the water cycle",
        "What is Newton's first law?",
        "How do hash functions work?",
        "What is a mutex in programming?",
        "Explain photosynthesis",
        "What is the speed of sound?",
    ],
    "negative": [
        # Current events, real-time data, recent news
        "What is the current price of gold?",
        "Who won the election yesterday?",
        "What's the weather in Paris right now?",
        "Latest news about SpaceX launch",
        "Current COVID case numbers",
        "Today's stock market performance",
        "Recent developments in Ukraine",
        "What time does the store close today?",
        "Is the restaurant open now?",
        "What movies are playing tonight?",
        "Current traffic conditions on I-95",
        "Latest iPhone release date",
        "Today's exchange rate for EUR to USD",
        "Current score of the basketball game",
        "Recent announcements from Apple",
    ],
}

MEMORY_REFERENCE_CALIBRATION = {
    "positive": [
        # References to prior conversation
        "What did I say before?",
        "Can you remind me of our discussion?",
        "You told me something earlier",
        "Go back to what we agreed on",
        "What was my preference again?",
        "Earlier in this chat you mentioned",
        "Recall what I asked you to remember",
        "What did you suggest last time?",
        "In our previous exchange",
        "What were the requirements I gave you?",
        "You recommended something before",
        "What was that code snippet you showed me?",
        "Go back to our earlier topic",
        "What settings did I choose?",
        "Remind me of the steps we discussed",
    ],
    "negative": [
        # General questions, not about prior conversation
        "What is the best Python library for ML?",
        "How do I install Node.js?",
        "Search for restaurants in Seattle",
        "What is the weather forecast?",
        "Find documentation for React hooks",
        "Calculate 15% tip on $50",
        "Translate hello to Spanish",
        "What time is it in Tokyo?",
        "Compare AWS vs GCP pricing",
        "How to fix this error message",
        "Best practices for API design",
        "Show me examples of async/await",
        "What is the syntax for list comprehension?",
        "Generate a random password",
        "How do I center a div in CSS?",
    ],
}


def run_calibration():
    """Run calibration and save results."""
    print("Loading semantic classifier...")
    classifier = SemanticClassifier()

    results = {}

    # Calibrate static knowledge threshold
    print("\nCalibrating static_knowledge threshold...")
    static_result = classifier.cross_validate_threshold(
        positive_examples=STATIC_KNOWLEDGE_CALIBRATION["positive"],
        negative_examples=STATIC_KNOWLEDGE_CALIBRATION["negative"],
        category="static_knowledge",
        k_folds=5,
        threshold_range=(0.30, 0.60),
        threshold_steps=30,
    )
    results["static_knowledge"] = {
        "threshold": round(static_result.optimal_threshold, 3),
        "mean_accuracy": round(static_result.mean_accuracy, 3),
        "std_accuracy": round(static_result.std_accuracy, 3),
        "auc": round(static_result.auc, 3) if static_result.auc else None,
        "fold_results": static_result.fold_results,
    }
    print(f"  Optimal threshold: {static_result.optimal_threshold:.3f}")
    print(f"  Mean accuracy: {static_result.mean_accuracy:.3f} (+/- {static_result.std_accuracy:.3f})")
    print(f"  AUC: {static_result.auc:.3f}")

    # Calibrate memory reference threshold
    print("\nCalibrating memory_reference threshold...")
    memory_result = classifier.cross_validate_threshold(
        positive_examples=MEMORY_REFERENCE_CALIBRATION["positive"],
        negative_examples=MEMORY_REFERENCE_CALIBRATION["negative"],
        category="memory_reference",
        k_folds=5,
        threshold_range=(0.30, 0.60),
        threshold_steps=30,
    )
    results["memory_reference"] = {
        "threshold": round(memory_result.optimal_threshold, 3),
        "mean_accuracy": round(memory_result.mean_accuracy, 3),
        "std_accuracy": round(memory_result.std_accuracy, 3),
        "auc": round(memory_result.auc, 3) if memory_result.auc else None,
        "fold_results": memory_result.fold_results,
    }
    print(f"  Optimal threshold: {memory_result.optimal_threshold:.3f}")
    print(f"  Mean accuracy: {memory_result.mean_accuracy:.3f} (+/- {memory_result.std_accuracy:.3f})")
    print(f"  AUC: {memory_result.auc:.3f}")

    # Duplicate search uses a fixed high threshold (similarity-based, not category-based)
    results["duplicate_search"] = {
        "threshold": 0.80,
        "note": "Duplicate search uses direct similarity, not cross-validated",
    }

    # Save results
    output_path = Path(__file__).parent / "thresholds.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nCalibration saved to {output_path}")

    # Also save sensitivity analysis
    sensitivity = analyze_threshold_sensitivity(classifier)
    sensitivity_path = Path(__file__).parent / "sensitivity_analysis.json"
    with open(sensitivity_path, "w") as f:
        json.dump(sensitivity, f, indent=2)
    print(f"Sensitivity analysis saved to {sensitivity_path}")

    return results


def analyze_threshold_sensitivity(classifier: SemanticClassifier) -> dict:
    """Analyze how results change with threshold variations of +/- 0.05."""
    sensitivity = {}

    for category, data in [
        ("static_knowledge", STATIC_KNOWLEDGE_CALIBRATION),
        ("memory_reference", MEMORY_REFERENCE_CALIBRATION),
    ]:
        # Get current threshold
        if category == "static_knowledge":
            base_threshold = classifier.thresholds.static_knowledge
        else:
            base_threshold = classifier.thresholds.memory_reference

        variations = []
        for delta in [-0.10, -0.05, 0.0, 0.05, 0.10]:
            threshold = base_threshold + delta

            # Compute accuracy at this threshold
            pos_embeddings = classifier.model.encode(data["positive"])
            neg_embeddings = classifier.model.encode(data["negative"])

            if category == "static_knowledge":
                centroid = classifier._static_centroid
            else:
                centroid = classifier._memory_centroid

            pos_sims = [classifier._cosine_similarity(e, centroid) for e in pos_embeddings]
            neg_sims = [classifier._cosine_similarity(e, centroid) for e in neg_embeddings]

            # True positives: positive examples above threshold
            tp = sum(1 for s in pos_sims if s >= threshold)
            # True negatives: negative examples below threshold
            tn = sum(1 for s in neg_sims if s < threshold)
            # False positives: negative examples above threshold
            fp = sum(1 for s in neg_sims if s >= threshold)
            # False negatives: positive examples below threshold
            fn = sum(1 for s in pos_sims if s < threshold)

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            accuracy = (tp + tn) / (tp + tn + fp + fn)
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

            variations.append({
                "threshold": round(threshold, 3),
                "delta": round(delta, 2),
                "accuracy": round(accuracy, 3),
                "precision": round(precision, 3),
                "recall": round(recall, 3),
                "f1": round(f1, 3),
                "true_positives": tp,
                "true_negatives": tn,
                "false_positives": fp,
                "false_negatives": fn,
            })

        sensitivity[category] = {
            "base_threshold": round(base_threshold, 3),
            "variations": variations,
        }

    return sensitivity


if __name__ == "__main__":
    run_calibration()
