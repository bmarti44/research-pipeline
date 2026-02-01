"""
Unified exemplars for semantic classification.

IMPORTANT: Data Leakage Prevention
==================================
The exemplars in this file are used to compute centroids for semantic classification.
To avoid data leakage, these exemplars MUST be kept separate from test scenarios:

1. CALIBRATION EXEMPLARS (this file): Used to train/calibrate the semantic classifier
2. TEST SCENARIOS (scenarios/generator.py): Used to evaluate the classifier

Rule: If a query appears in this file, it MUST NOT appear in test scenarios, and vice versa.

The test scenarios in scenarios/generator.py have been carefully designed to test the
same concepts WITHOUT using the exact same queries. For example:
- Exemplar: "What is the capital of France?"
- Test: "What is the boiling point of water?" (different query, same static-knowledge pattern)

When adding new exemplars or test scenarios, ensure no overlap exists.
"""

# ============================================================================
# CALIBRATION EXEMPLARS - DO NOT USE IN TEST SCENARIOS
# ============================================================================

STATIC_KNOWLEDGE_EXEMPLARS = [
    # Basic facts (used for calibration only)
    "What is the capital of France?",
    "What is photosynthesis?",
    "Define recursion",
    "Explain how inheritance works in OOP",
    "What is the Pythagorean theorem?",
    "Who wrote Romeo and Juliet?",
    "What is the speed of light?",
    "Explain the theory of relativity",
    # Programming concepts
    "What is a linked list?",
    "How does garbage collection work?",
    "What is the difference between a class and an object?",
    "Explain big O notation",
    "What is a hash table?",
    "How does TCP/IP work?",
    # Science
    "What causes earthquakes?",
    "How do vaccines work?",
    "What is DNA?",
    "Explain how evolution works",
    # Math
    "What is a prime number?",
    "Explain the quadratic formula",
    "What is calculus used for?",
    # General knowledge
    "Who was Albert Einstein?",
    "What is democracy?",
    "How does the stock market work?",
]

MEMORY_REFERENCE_EXEMPLARS = [
    # Memory reference patterns (used for calibration only)
    "What did we talk about yesterday?",
    "Remember when we discussed the project?",
    "You mentioned something about deadlines earlier",
    "What did I tell you about my preferences?",
    "In our last conversation you said",
    "We talked about this before",
    "What was my budget again?",
    "As I mentioned previously",
    # More variations
    "Earlier you told me",
    "Go back to what we discussed",
    "What was that thing you said?",
    "Remind me what we decided",
    "You suggested something before",
    "What did we agree on?",
    "Can you recall our earlier discussion?",
    "What was my original request?",
]

# ============================================================================
# HELD-OUT VALIDATION SET - For testing calibration without data leakage
# ============================================================================
# These are NOT used for computing centroids. Use for validation only.

STATIC_KNOWLEDGE_HOLDOUT = [
    # Held-out examples for validation (NOT in training set)
    "What is the boiling point of water?",
    "How do binary trees work?",
    "What is the mitochondria's function?",
    "Who painted the Mona Lisa?",
    "What is Euler's formula?",
]

MEMORY_REFERENCE_HOLDOUT = [
    # Held-out examples for validation (NOT in training set)
    "What were we chatting about last week?",
    "Recall our conversation about the deadline?",
    "What was the price range I gave you?",
    "Earlier you noted something about the timeline",
]
