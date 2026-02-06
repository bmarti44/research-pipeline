"""
Generate Chain-of-Thought Training Data for COCONUT

Uses Claude API to generate math problems with step-by-step reasoning.
This creates the training data needed for curriculum training.
"""

import json
import os
import random
import asyncio
from pathlib import Path
from typing import List, Dict, Any

# FIX R4: Make anthropic import optional for synthetic data generation
try:
    import anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False


# Problem templates for various difficulty levels
PROBLEM_TEMPLATES = {
    "arithmetic": [
        "Calculate: {a} + {b} × {c}",
        "What is {a} × {b} - {c}?",
        "Solve: ({a} + {b}) × {c}",
        "Calculate: {a} ÷ {b} + {c}",
        "What is {a}² + {b}?",
    ],
    "word_problems": [
        "Sarah has {a} apples. She buys {b} more and gives away {c}. How many does she have?",
        "A train travels {a} miles in {b} hours. What is its average speed?",
        "If {a} workers can finish a job in {b} days, how many days for {c} workers?",
        "A store has {a} items. After selling {b}% of them, how many remain?",
        "Tom is {a} years old. In {b} years, he will be twice as old as he was {c} years ago. Is this true?",
    ],
    "algebra": [
        "Solve for x: {a}x + {b} = {c}",
        "If 2x + {a} = {b}, what is x?",
        "Solve: x² = {a}",
        "What value of x satisfies: {a}x = {b}?",
        "If x + y = {a} and x - y = {b}, what is x?",
    ],
    "logical": [
        "If all A are B, and all B are C, and X is an A, is X a C?",
        "There are {a} people in a room. If {b} leave and {c} enter, how many are there?",
        "A is taller than B. B is taller than C. Who is shortest?",
        "If today is Monday and the meeting is in {a} days, what day is the meeting?",
        "In a sequence where each term is the previous term plus {a}, starting from {b}, what is the {c}th term?",
    ],
}


def generate_problem(category: str, difficulty: int = 1) -> Dict[str, Any]:
    """Generate a random problem from templates."""
    template = random.choice(PROBLEM_TEMPLATES[category])

    # Scale numbers by difficulty
    max_val = 10 * difficulty
    a = random.randint(1, max_val)
    b = random.randint(1, max_val)
    c = random.randint(1, max_val)

    problem = template.format(a=a, b=b, c=c)
    return {"question": problem, "category": category, "difficulty": difficulty, "params": {"a": a, "b": b, "c": c}}


async def generate_cot_solution(client: Any, problem: Dict[str, Any]) -> Dict[str, Any]:
    """Use Claude to generate step-by-step solution."""
    prompt = f"""Solve this problem step by step. Format your response as JSON with:
- "steps": list of reasoning steps (each step should be a short sentence)
- "answer": final numeric or text answer

Problem: {problem['question']}

Respond ONLY with valid JSON, no other text."""

    try:
        response = await client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=500,
            messages=[{"role": "user", "content": prompt}]
        )

        content = response.content[0].text.strip()
        # Handle potential markdown code blocks
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]

        solution = json.loads(content)
        return {
            "question": problem["question"],
            "steps": solution.get("steps", []),
            "answer": str(solution.get("answer", "")),
            "category": problem["category"],
            "difficulty": problem["difficulty"],
        }
    except Exception as e:
        print(f"Error generating solution: {e}")
        return None


async def generate_dataset(
    n_samples: int = 1000,
    output_path: str = "cot_training_data.json",
    categories: List[str] = None,
) -> List[Dict[str, Any]]:
    """Generate full CoT dataset."""
    if categories is None:
        categories = list(PROBLEM_TEMPLATES.keys())

    client = anthropic.AsyncAnthropic()

    print(f"Generating {n_samples} CoT samples...")

    # Generate problems
    problems = []
    for i in range(n_samples):
        category = random.choice(categories)
        difficulty = random.randint(1, 3)
        problems.append(generate_problem(category, difficulty))

    # Generate solutions in batches
    batch_size = 10
    all_samples = []

    for i in range(0, len(problems), batch_size):
        batch = problems[i:i + batch_size]
        tasks = [generate_cot_solution(client, p) for p in batch]
        results = await asyncio.gather(*tasks)

        for result in results:
            if result and result.get("steps"):
                all_samples.append(result)

        print(f"Progress: {min(i + batch_size, len(problems))}/{len(problems)}")

    # Save to file
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w") as f:
        json.dump(all_samples, f, indent=2)

    print(f"Generated {len(all_samples)} samples, saved to {output_path}")
    return all_samples


def generate_synthetic_cot_data(n_samples: int = 500, seed: int = 42) -> List[Dict[str, Any]]:
    """
    Generate synthetic CoT data without API calls.
    Uses deterministic problems with programmatic solutions.

    Args:
        n_samples: Number of samples to generate
        seed: Random seed for reproducibility (FIX: added for determinism)
    """
    # FIX: Set seed for reproducibility
    random.seed(seed)

    samples = []

    for i in range(n_samples):
        problem_type = random.choice(["addition", "multiplication", "sequence", "comparison"])

        if problem_type == "addition":
            a, b = random.randint(1, 100), random.randint(1, 100)
            question = f"What is {a} + {b}?"
            steps = [
                f"I need to add {a} and {b}.",
                f"Starting with {a}.",
                f"Adding {b} to get {a + b}.",
            ]
            answer = str(a + b)

        elif problem_type == "multiplication":
            a, b = random.randint(1, 20), random.randint(1, 20)
            question = f"What is {a} × {b}?"
            steps = [
                f"I need to multiply {a} by {b}.",
                f"This means adding {a} to itself {b} times.",
                f"The result is {a * b}.",
            ]
            answer = str(a * b)

        elif problem_type == "sequence":
            start = random.randint(1, 10)
            step = random.randint(1, 5)
            n = random.randint(3, 7)
            question = f"In a sequence starting at {start} with step {step}, what is the {n}th term?"
            result = start + (n - 1) * step
            steps = [
                f"The sequence starts at {start}.",
                f"Each term increases by {step}.",
                f"The {n}th term is {start} + ({n}-1) × {step}.",
                f"That equals {start} + {(n-1) * step} = {result}.",
            ]
            answer = str(result)

        else:  # comparison
            a, b, c = random.randint(1, 100), random.randint(1, 100), random.randint(1, 100)
            question = f"Which is largest: {a}, {b}, or {c}?"
            largest = max(a, b, c)
            steps = [
                f"I need to compare {a}, {b}, and {c}.",
                f"Comparing {a} and {b}: {max(a, b)} is larger.",
                f"Comparing {max(a, b)} and {c}: {largest} is largest.",
            ]
            answer = str(largest)

        samples.append({
            "question": question,
            "steps": steps,
            "answer": answer,
            "category": problem_type,
            "difficulty": 1,
            "idx": i,
        })

    return samples


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--n_samples", type=int, default=500)
    parser.add_argument("--output", type=str, default="data/cot_training_data.json")
    parser.add_argument("--use_api", action="store_true", help="Use Claude API (requires ANTHROPIC_API_KEY)")
    args = parser.parse_args()

    if args.use_api and os.environ.get("ANTHROPIC_API_KEY") and HAS_ANTHROPIC:
        asyncio.run(generate_dataset(args.n_samples, args.output))
    elif args.use_api and not HAS_ANTHROPIC:
        print("WARNING: --use_api specified but anthropic module not installed")
        print("Install with: pip install anthropic")
        print("Falling back to synthetic data generation...")
        samples = generate_synthetic_cot_data(args.n_samples)
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(samples, f, indent=2)
        print(f"Generated {len(samples)} synthetic samples, saved to {args.output}")
    else:
        print("Generating synthetic CoT data (no API)...")
        samples = generate_synthetic_cot_data(args.n_samples)

        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(samples, f, indent=2)

        print(f"Generated {len(samples)} synthetic samples, saved to {args.output}")
