"""Generate OOD test data in Meta's ProsQA format.

Uses the EXACT same vocabulary as ProsQA training data to test
reasoning generalization, not vocabulary generalization.

Produces 4 OOD test sets:
1. ood_7hop.json  - 1K samples with 7-hop reasoning chains (ProsQA max is 6)
2. ood_8hop.json  - 1K samples with 8-hop reasoning chains
3. ood_dag.json   - 1K samples with DAG structure (convergent paths, 4 hops)
4. ood_dense.json - 1K samples with dense graphs (high branching, 4 hops)
"""

import json
import random

# Exact vocabulary from ProsQA training data
PERSON_NAMES = [
    "Alex", "Bob", "Carol", "Davis", "Eva", "Fae", "Gabriel",
    "Jack", "Max", "Oliver", "Polly", "Rex", "Sally", "Sam",
    "Stella", "Tom", "Wren",
]

SPECIES_NAMES = [
    "bompus", "boompus", "brimpus", "chorpus", "dumpus", "felpus",
    "fompus", "gerpus", "gorpus", "grimpus", "gwompus", "hilpus",
    "impus", "jelpus", "jompus", "kerpus", "lempus", "lorpus",
    "numpus", "quimpus", "rempus", "rompus", "rorpus", "scrompus",
    "shumpus", "sterpus", "storpus", "terpus", "timpus", "tumpus",
    "vumpus", "worpus", "wumpus", "yerpus", "yimpus", "yumpus",
    "zhorpus", "zumpus",
]


def generate_tree_sample(n_hops, rng):
    """Generate a ProsQA-format sample with a tree graph."""
    person = rng.choice(PERSON_NAMES)

    # Sample species for the chain + distractors
    # Need n_hops+1 for chain, plus distractors
    n_needed = min(n_hops + 1 + rng.randint(5, 12), len(SPECIES_NAMES))
    species = rng.sample(SPECIES_NAMES, n_needed)

    # idx_to_symbol: [person, species...]
    idx_to_symbol = [person] + species
    root_idx = 0

    # Chain: indices 1 through n_hops+1
    chain = list(range(1, n_hops + 2))
    target_idx = chain[-1]

    # Main chain edges
    edges = []
    for i in range(len(chain) - 1):
        edges.append([chain[i], chain[i + 1]])

    # Pick neg_target from species not on the chain
    distractor_indices = list(range(n_hops + 2, len(idx_to_symbol)))
    neg_target_idx = rng.choice(distractor_indices)

    # Add distractor edges
    for _ in range(rng.randint(n_hops * 2, n_hops * 4)):
        src = rng.choice(list(range(len(idx_to_symbol))))
        dst = rng.choice(distractor_indices)
        if src != dst and [src, dst] not in edges:
            edges.append([src, dst])

    return _build_sample(
        person, idx_to_symbol, edges, chain, root_idx, target_idx,
        neg_target_idx, rng
    )


def generate_dag_sample(n_hops, rng):
    """Generate a sample with DAG structure (multiple convergent paths)."""
    person = rng.choice(PERSON_NAMES)

    n_needed = min(n_hops + 1 + rng.randint(6, 14), len(SPECIES_NAMES))
    species = rng.sample(SPECIES_NAMES, n_needed)

    idx_to_symbol = [person] + species
    root_idx = 0

    chain = list(range(1, n_hops + 2))
    target_idx = chain[-1]

    edges = []
    for i in range(len(chain) - 1):
        edges.append([chain[i], chain[i + 1]])

    # Add convergent paths to intermediate chain nodes
    alt_start = n_hops + 2
    for conv_point in range(2, min(n_hops + 1, 5)):
        alt_idx = alt_start
        alt_start += 1
        if alt_idx < len(idx_to_symbol):
            edges.append([chain[0], alt_idx])
            edges.append([alt_idx, chain[conv_point]])

    distractor_indices = list(range(alt_start, len(idx_to_symbol)))
    neg_target_idx = rng.choice(distractor_indices) if distractor_indices else alt_start - 1

    for _ in range(rng.randint(n_hops * 2, n_hops * 3)):
        src = rng.choice(list(range(len(idx_to_symbol))))
        dst_pool = distractor_indices if distractor_indices else list(range(1, len(idx_to_symbol)))
        dst = rng.choice(dst_pool)
        if src != dst and [src, dst] not in edges:
            edges.append([src, dst])

    return _build_sample(
        person, idx_to_symbol, edges, chain, root_idx, target_idx,
        neg_target_idx, rng
    )


def generate_dense_sample(n_hops, rng):
    """Generate a sample with dense graph (high branching factor)."""
    person = rng.choice(PERSON_NAMES)

    # Use all available species for maximum density
    n_needed = min(n_hops + 1 + rng.randint(10, 20), len(SPECIES_NAMES))
    species = rng.sample(SPECIES_NAMES, n_needed)

    idx_to_symbol = [person] + species
    root_idx = 0

    chain = list(range(1, n_hops + 2))
    target_idx = chain[-1]

    edges = []
    for i in range(len(chain) - 1):
        edges.append([chain[i], chain[i + 1]])

    # High branching: each chain node connects to 3-6 non-chain nodes
    non_chain = list(range(n_hops + 2, len(idx_to_symbol)))
    rng.shuffle(non_chain)
    branch_idx = 0

    for chain_node in chain:
        remaining = len(non_chain) - branch_idx
        if remaining < 1:
            break
        n_branches = rng.randint(1, min(6, remaining))
        for _ in range(n_branches):
            if branch_idx < len(non_chain):
                edges.append([chain_node, non_chain[branch_idx]])
                branch_idx += 1

    # Cross-connections between non-chain nodes
    for _ in range(len(non_chain)):
        src = rng.choice(non_chain[:max(1, branch_idx)])
        dst = rng.choice(non_chain)
        if src != dst and [src, dst] not in edges:
            edges.append([src, dst])

    available_neg = [n for n in non_chain if n != target_idx]
    neg_target_idx = rng.choice(available_neg) if available_neg else non_chain[0]

    return _build_sample(
        person, idx_to_symbol, edges, chain, root_idx, target_idx,
        neg_target_idx, rng
    )


def _build_sample(person, idx_to_symbol, edges, chain, root_idx,
                   target_idx, neg_target_idx, rng):
    """Build a ProsQA-format sample dict from graph components."""
    # Build fact lines (randomized order, matching ProsQA format)
    shuffled_edges = edges.copy()
    rng.shuffle(shuffled_edges)

    fact_lines = []
    for src, dst in shuffled_edges:
        src_name = idx_to_symbol[src]
        dst_name = idx_to_symbol[dst]
        if src == root_idx:
            fact_lines.append(f"{src_name} is a {dst_name}.")
        else:
            fact_lines.append(f"Every {src_name} is a {dst_name}.")

    target_name = idx_to_symbol[target_idx]
    neg_target_name = idx_to_symbol[neg_target_idx]

    # Randomize option order in question
    if rng.random() < 0.5:
        option_str = f"a {target_name} or a {neg_target_name}"
    else:
        option_str = f"a {neg_target_name} or a {target_name}"

    question = " ".join(fact_lines) + f" Is {person} {option_str}?"

    # Build reasoning steps
    steps = [f"{person} is a {idx_to_symbol[chain[0]]}."]
    for i in range(len(chain) - 1):
        from_name = idx_to_symbol[chain[i]]
        to_name = idx_to_symbol[chain[i + 1]]
        steps.append(
            f"Every {from_name} is a {to_name}, so {person} is a {to_name}."
        )

    answer = f"{person} is a {target_name}."

    return {
        "question": question,
        "steps": steps,
        "answer": answer,
        "idx_to_symbol": idx_to_symbol,
        "edges": edges,
        "root": root_idx,
        "target": target_idx,
        "neg_target": neg_target_idx,
    }


def main():
    rng = random.Random(42)

    configs = [
        ("ood_7hop.json",  1000, 7, "tree"),
        ("ood_8hop.json",  1000, 8, "tree"),
        ("ood_dag.json",   1000, 4, "dag"),
        ("ood_dense.json", 1000, 4, "dense"),
    ]

    for filename, n_samples, n_hops, graph_type in configs:
        gen_fn = {
            "tree": generate_tree_sample,
            "dag": generate_dag_sample,
            "dense": generate_dense_sample,
        }[graph_type]

        samples = [gen_fn(n_hops, rng) for _ in range(n_samples)]

        outpath = f"data/{filename}"
        with open(outpath, "w") as f:
            json.dump(samples, f)

        avg_steps = sum(len(s["steps"]) for s in samples) / len(samples)
        avg_edges = sum(len(s["edges"]) for s in samples) / len(samples)
        avg_qlen = sum(len(s["question"]) for s in samples) / len(samples)
        print(f"{filename}: {n_samples} samples, {n_hops} hops ({graph_type})")
        print(f"  avg steps: {avg_steps:.1f}, avg edges: {avg_edges:.1f}, "
              f"avg question chars: {avg_qlen:.0f}")


if __name__ == "__main__":
    main()
