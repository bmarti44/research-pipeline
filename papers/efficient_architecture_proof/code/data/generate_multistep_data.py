"""
Generate multi-step word problems for A+B+C study (v3).

Addresses review feedback: "Data too simple (arithmetic instead of complex reasoning)"

Creates problems that require 4-6 reasoning steps, not just 2-3.
Categories:
1. Multi-step word problems with intermediate calculations
2. Sequential reasoning (if A then B then C)
3. Problems requiring back-tracking or state updates
"""

import json
import random
from pathlib import Path
from typing import List, Dict


def generate_shopping_problem() -> Dict:
    """Multi-step shopping problem with running total."""
    items = [
        ("apple", random.randint(1, 5)),
        ("banana", random.randint(1, 5)),
        ("orange", random.randint(1, 5)),
        ("milk", random.randint(2, 8)),
        ("bread", random.randint(2, 6)),
    ]

    # Pick 3-4 items
    selected = random.sample(items, random.randint(3, 4))
    quantities = [random.randint(1, 3) for _ in selected]
    budget = sum(q * p for (_, p), q in zip(selected, quantities)) + random.randint(5, 15)

    question = f"Sarah has ${budget}. She buys "
    parts = [f"{q} {item}s at ${price} each" for (item, price), q in zip(selected, quantities)]
    question += ", ".join(parts[:-1]) + f", and {parts[-1]}. How much money does she have left?"

    steps = []
    total = 0
    for (item, price), q in zip(selected, quantities):
        cost = q * price
        total += cost
        steps.append(f"Cost of {q} {item}s: {q} × ${price} = ${cost}.")
        steps.append(f"Running total: ${total}.")

    remaining = budget - total
    steps.append(f"Money left: ${budget} - ${total} = ${remaining}.")

    return {
        "question": question,
        "steps": steps,
        "answer": str(remaining),
        "category": "shopping",
        "n_steps": len(steps),
    }


def generate_time_problem() -> Dict:
    """Multi-step time calculation problem."""
    start_hour = random.randint(6, 10)

    activities = [
        ("breakfast", random.randint(15, 45)),
        ("commute to work", random.randint(20, 60)),
        ("morning meeting", random.randint(30, 90)),
        ("lunch break", random.randint(30, 60)),
        ("afternoon work", random.randint(60, 180)),
    ]

    selected = random.sample(activities, random.randint(3, 5))

    question = f"Tom starts his day at {start_hour}:00 AM. "
    parts = [f"{act} takes {mins} minutes" for act, mins in selected]
    question += ", ".join(parts) + ". What time does he finish?"

    steps = []
    total_mins = start_hour * 60
    for act, mins in selected:
        total_mins += mins
        hours = total_mins // 60
        mins_part = total_mins % 60
        am_pm = "AM" if hours < 12 else "PM"
        display_hour = hours if hours <= 12 else hours - 12
        if display_hour == 0:
            display_hour = 12
        steps.append(f"After {act} (+{mins} min): {display_hour}:{mins_part:02d} {am_pm}.")

    final_hours = total_mins // 60
    final_mins = total_mins % 60
    am_pm = "AM" if final_hours < 12 else "PM"
    display_hour = final_hours if final_hours <= 12 else final_hours - 12
    if display_hour == 0:
        display_hour = 12
    answer = f"{display_hour}:{final_mins:02d} {am_pm}"

    return {
        "question": question,
        "steps": steps,
        "answer": answer,
        "category": "time",
        "n_steps": len(steps),
    }


def generate_distance_problem() -> Dict:
    """Multi-step distance/speed problem."""
    segments = random.randint(3, 4)
    speeds = [random.choice([20, 30, 40, 50, 60]) for _ in range(segments)]
    times = [random.randint(1, 3) for _ in range(segments)]

    modes = ["walking", "biking", "driving", "running"]
    selected_modes = random.sample(modes, segments)

    question = f"Alex travels by "
    parts = [f"{mode} at {speed} mph for {time} hours" for mode, speed, time in zip(selected_modes, speeds, times)]
    question += ", then ".join(parts) + ". What is the total distance traveled?"

    steps = []
    total = 0
    for mode, speed, time in zip(selected_modes, speeds, times):
        dist = speed * time
        total += dist
        steps.append(f"Distance {mode}: {speed} mph × {time} hours = {dist} miles.")
        steps.append(f"Total so far: {total} miles.")

    steps.append(f"Total distance: {total} miles.")

    return {
        "question": question,
        "steps": steps,
        "answer": f"{total} miles",
        "category": "distance",
        "n_steps": len(steps),
    }


def generate_inventory_problem() -> Dict:
    """Multi-step inventory tracking with additions and removals."""
    initial = random.randint(50, 200)
    item = random.choice(["books", "widgets", "boxes", "packages"])

    operations = []
    for _ in range(random.randint(3, 5)):
        if random.random() < 0.5:
            amt = random.randint(10, 50)
            operations.append(("received", amt))
        else:
            amt = random.randint(5, 30)
            operations.append(("shipped", amt))

    question = f"A warehouse starts with {initial} {item}. "
    parts = []
    for op, amt in operations:
        if op == "received":
            parts.append(f"receives {amt}")
        else:
            parts.append(f"ships out {amt}")
    question += ", ".join(parts) + f". How many {item} remain?"

    steps = [f"Starting inventory: {initial} {item}."]
    current = initial
    for op, amt in operations:
        if op == "received":
            current += amt
            steps.append(f"After receiving {amt}: {current} {item}.")
        else:
            current -= amt
            steps.append(f"After shipping {amt}: {current} {item}.")

    # Ensure non-negative
    if current < 0:
        return generate_inventory_problem()  # Regenerate

    return {
        "question": question,
        "steps": steps,
        "answer": str(current),
        "category": "inventory",
        "n_steps": len(steps),
    }


def generate_conditional_problem() -> Dict:
    """Problem with conditional reasoning."""
    score = random.randint(50, 100)
    thresholds = sorted(random.sample(range(60, 95, 5), 3))
    grades = ["A", "B", "C", "D"]

    question = f"In a class, scoring above {thresholds[2]} gets an A, "
    question += f"above {thresholds[1]} gets a B, above {thresholds[0]} gets a C, "
    question += f"otherwise D. If a student scores {score}, what grade do they get?"

    steps = [f"Student's score: {score}."]
    steps.append(f"Check if {score} > {thresholds[2]} (A threshold): {'Yes' if score > thresholds[2] else 'No'}.")

    if score > thresholds[2]:
        grade = "A"
    else:
        steps.append(f"Check if {score} > {thresholds[1]} (B threshold): {'Yes' if score > thresholds[1] else 'No'}.")
        if score > thresholds[1]:
            grade = "B"
        else:
            steps.append(f"Check if {score} > {thresholds[0]} (C threshold): {'Yes' if score > thresholds[0] else 'No'}.")
            if score > thresholds[0]:
                grade = "C"
            else:
                grade = "D"

    steps.append(f"Final grade: {grade}.")

    return {
        "question": question,
        "steps": steps,
        "answer": grade,
        "category": "conditional",
        "n_steps": len(steps),
    }


def generate_average_problem() -> Dict:
    """Problem requiring finding a missing value given average."""
    n_known = random.randint(3, 5)
    known_values = [random.randint(70, 100) for _ in range(n_known)]
    target_avg = random.randint(80, 95)

    total_needed = target_avg * (n_known + 1)
    unknown = total_needed - sum(known_values)

    names = ["Math", "Science", "English", "History", "Art", "Music"]
    selected = random.sample(names, n_known)

    question = f"A student scored "
    parts = [f"{score} in {subject}" for score, subject in zip(known_values, selected)]
    question += ", ".join(parts)
    question += f". What score do they need on the final exam to have an average of {target_avg}?"

    steps = [f"Known scores: {known_values}."]
    steps.append(f"Sum of known scores: {sum(known_values)}.")
    steps.append(f"Number of exams including final: {n_known + 1}.")
    steps.append(f"Total needed for average of {target_avg}: {target_avg} × {n_known + 1} = {total_needed}.")
    steps.append(f"Score needed on final: {total_needed} - {sum(known_values)} = {unknown}.")

    # Ensure reasonable score
    if unknown < 0 or unknown > 150:
        return generate_average_problem()  # Regenerate

    return {
        "question": question,
        "steps": steps,
        "answer": str(unknown),
        "category": "average",
        "n_steps": len(steps),
    }


def generate_percentage_chain() -> Dict:
    """Chained percentage calculations."""
    initial = random.randint(100, 500)

    operations = []
    for _ in range(random.randint(3, 4)):
        pct = random.choice([10, 15, 20, 25, 30])
        op = random.choice(["increase", "decrease"])
        operations.append((op, pct))

    question = f"A price starts at ${initial}. It "
    parts = [f"{'increases' if op == 'increase' else 'decreases'} by {pct}%" for op, pct in operations]
    question += ", then ".join(parts) + ". What is the final price?"

    steps = [f"Starting price: ${initial}."]
    current = initial
    for op, pct in operations:
        change = current * pct / 100
        if op == "increase":
            current = current + change
            steps.append(f"After {pct}% increase: ${current:.2f}.")
        else:
            current = current - change
            steps.append(f"After {pct}% decrease: ${current:.2f}.")

    final = round(current, 2)
    steps.append(f"Final price: ${final}.")

    return {
        "question": question,
        "steps": steps,
        "answer": f"${final}",
        "category": "percentage",
        "n_steps": len(steps),
    }


def generate_dataset(n_samples: int = 600, val_ratio: float = 0.15) -> tuple:
    """Generate train and validation datasets."""
    generators = [
        generate_shopping_problem,
        generate_time_problem,
        generate_distance_problem,
        generate_inventory_problem,
        generate_conditional_problem,
        generate_average_problem,
        generate_percentage_chain,
    ]

    samples = []
    for i in range(n_samples):
        gen = random.choice(generators)
        sample = gen()
        sample["idx"] = i
        samples.append(sample)

    # Shuffle and split
    random.shuffle(samples)
    n_val = int(n_samples * val_ratio)
    val_samples = samples[:n_val]
    train_samples = samples[n_val:]

    # Reindex
    for i, s in enumerate(train_samples):
        s["idx"] = i
    for i, s in enumerate(val_samples):
        s["idx"] = i

    return train_samples, val_samples


def main():
    random.seed(42)  # Reproducible

    output_dir = Path(__file__).parent

    # Generate datasets
    train_data, val_data = generate_dataset(n_samples=600, val_ratio=0.15)

    # Stats
    train_steps = [s["n_steps"] for s in train_data]
    val_steps = [s["n_steps"] for s in val_data]

    print(f"Generated {len(train_data)} training samples")
    print(f"Generated {len(val_data)} validation samples")
    print(f"Training steps per problem: {sum(train_steps)/len(train_steps):.1f} avg (range {min(train_steps)}-{max(train_steps)})")

    # Category distribution
    from collections import Counter
    cats = Counter(s["category"] for s in train_data)
    print("\nCategory distribution (train):")
    for cat, count in sorted(cats.items()):
        print(f"  {cat}: {count}")

    # Save
    with open(output_dir / "multistep_train.json", "w") as f:
        json.dump(train_data, f, indent=2)

    with open(output_dir / "multistep_val.json", "w") as f:
        json.dump(val_data, f, indent=2)

    print(f"\nSaved to {output_dir / 'multistep_train.json'}")
    print(f"Saved to {output_dir / 'multistep_val.json'}")

    # Show sample
    print("\n--- Sample Problem ---")
    sample = train_data[0]
    print(f"Q: {sample['question']}")
    print("Steps:")
    for step in sample["steps"]:
        print(f"  - {step}")
    print(f"A: {sample['answer']}")


if __name__ == "__main__":
    main()
