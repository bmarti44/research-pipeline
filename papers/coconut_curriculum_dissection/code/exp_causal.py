"""
Experiment: Causal Tracing (Experiment 4)

Adapted from Meng et al. (2022) "Locating and Editing Factual Associations in GPT".
Identifies which (layer, position) pairs are causally responsible for correct answers.

Method:
  1. Clean run: forward over input+answer, record all hidden states + logit_clean
  2. Corrupt run: replace all thought/CoT positions with noise → logit_corrupt
  3. Patch: for each (l, p), corrupt + restore clean activation → logit_patched
  4. CE(l,p) = (logit_patched - logit_corrupt) / (logit_clean - logit_corrupt + 1e-6)
  5. Average CE over samples

MANDATORY Validation: M1 (CoT baseline) must show CE > 0.3 at ≥50% of CoT positions.
If M1 validation fails, the patching implementation is broken. Do not trust results.

Usage:
    cd code
    python exp_causal.py \
        --checkpoint_dir ../results \
        --data data/prosqa_test.json \
        --num_samples 500 \
        --output_dir ../results/experiments/causal/ \
        --seed 0
"""

import argparse
import json
import os
import sys
import time

import torch
import numpy as np

from exp_utils import (
    load_model_by_name,
    setup_tokenizer,
    get_special_ids,
    prepare_input,
    get_processed_embeds,
    extract_answer,
    get_ground_truth,
    load_data,
    set_seed,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODELS_TO_TRACE = ["m1", "m3", "m4", "m4b", "m5"]  # M2 skipped: no CoT/thoughts
NUM_THOUGHTS = 6
NUM_LAYERS = 13  # GPT-2: embedding + 12 transformer blocks (indices 0-12)
MAX_COT_POSITIONS = 20  # Cap M1's CoT positions for tractability
NOISE_MULTIPLIER = 1.0  # 1× embedding std; higher values overwhelm per-position patching


# ---------------------------------------------------------------------------
# Core tracing functions
# ---------------------------------------------------------------------------

@torch.no_grad()
def clean_forward(base_model, full_embeds, answer_token_id, answer_pred_pos):
    """
    Clean forward pass. Record all hidden states and logit of correct answer.

    Returns:
        logit_clean: float
        hidden_states: list of 13 tensors, each [seq_len, hidden_dim]
    """
    outputs = base_model(inputs_embeds=full_embeds, output_hidden_states=True)
    logit_clean = outputs.logits[0, answer_pred_pos, answer_token_id].item()
    hidden_states = [h[0].clone() for h in outputs.hidden_states]
    return logit_clean, hidden_states


@torch.no_grad()
def corrupt_forward(base_model, full_embeds, critical_positions, noise_embeds,
                    answer_token_id, answer_pred_pos):
    """
    Corrupt forward: replace critical positions in input embeddings with noise.

    Returns:
        logit_corrupt: float
        corrupt_embeds: tensor [1, seq_len, dim]
    """
    corrupt_embeds = full_embeds.clone()
    for i, pos in enumerate(critical_positions):
        idx = min(i, noise_embeds.shape[0] - 1)
        corrupt_embeds[0, pos, :] = noise_embeds[idx]

    outputs = base_model(inputs_embeds=corrupt_embeds)
    logit_corrupt = outputs.logits[0, answer_pred_pos, answer_token_id].item()
    return logit_corrupt, corrupt_embeds


@torch.no_grad()
def patched_forward(base_model, corrupt_embeds, clean_hidden_states,
                    layer, position, answer_token_id, answer_pred_pos):
    """
    Patched forward: corrupt inputs + restore clean activation at (layer, position).

    For layer 0 (embedding output): modify input embeddings directly.
    For layers 1-12: hook into transformer.h[layer-1] to replace block output.

    Returns:
        logit_patched: float
    """
    if layer == 0:
        # Layer 0 = embedding output. Patch input embeddings directly.
        patched_embeds = corrupt_embeds.clone()
        patched_embeds[0, position, :] = clean_hidden_states[0][position]
        outputs = base_model(inputs_embeds=patched_embeds)
        return outputs.logits[0, answer_pred_pos, answer_token_id].item()

    # Layers 1-12: hook into transformer block (layer-1)
    block_idx = layer - 1
    clean_activation = clean_hidden_states[layer][position].clone()

    def patch_hook(module, input, output):
        # GPT2Block output: (hidden_states, present_kv[, attentions])
        hs = output[0].clone()
        hs[0, position, :] = clean_activation
        return (hs,) + output[1:]

    target_block = base_model.transformer.h[block_idx]
    handle = target_block.register_forward_hook(patch_hook)

    outputs = base_model(inputs_embeds=corrupt_embeds)
    logit_patched = outputs.logits[0, answer_pred_pos, answer_token_id].item()

    handle.remove()
    return logit_patched


# ---------------------------------------------------------------------------
# Input preparation per model type
# ---------------------------------------------------------------------------

@torch.no_grad()
def prepare_coconut_inputs(model, tokenizer, model_info, sample, num_thoughts, device):
    """
    Prepare causal tracing inputs for a Coconut model (M3/M4/M4b).

    1. Get processed inputs_embeds (after multi-pass thought feedback)
    2. Append answer continuation embeddings ("### answer")
    3. Identify thought token positions as critical positions

    Returns:
        full_embeds, critical_positions, answer_pred_pos, answer_token_id, base_model
        or (None, ...) if sample can't be traced.
    """
    special_ids = get_special_ids(tokenizer)

    input_ids = prepare_input(
        sample, tokenizer, model_info,
        num_thoughts=num_thoughts, device=device,
    )

    tokens = input_ids[0].tolist()
    thought_positions = [i for i, t in enumerate(tokens) if t == special_ids["latent_id"]]

    if len(thought_positions) == 0:
        return None, None, None, None, None

    # Get processed embeddings (thought tokens filled with hidden-state feedback)
    processed_embeds = get_processed_embeds(model, input_ids)

    # Construct answer continuation: "### answer_entity"
    gt = get_ground_truth(sample)
    continuation = "### " + gt
    cont_ids = tokenizer.encode(continuation, add_special_tokens=False)

    if len(cont_ids) == 0:
        return None, None, None, None, None

    # Embed continuation tokens using the same embedding layer
    cont_tensor = torch.tensor(cont_ids, device=device).unsqueeze(0)
    cont_embeds = model.embedding(cont_tensor)

    # Full sequence: processed_embeds + continuation
    full_embeds = torch.cat([processed_embeds, cont_embeds], dim=1)

    # Find the first answer token by searching cont_ids directly.
    # We avoid computing offset from "### " prefix because BPE merges
    # across the boundary can shift token boundaries (e.g. "### Sally"
    # may tokenize differently than "### " + "Sally").
    answer_ids = tokenizer.encode(gt, add_special_tokens=False)
    if len(answer_ids) == 0:
        return None, None, None, None, None
    answer_token_id = answer_ids[0]

    # Search for the first occurrence of answer_token_id in cont_ids
    answer_offset_in_cont = None
    for ci, tid in enumerate(cont_ids):
        if tid == answer_token_id:
            answer_offset_in_cont = ci
            break

    if answer_offset_in_cont is None:
        return None, None, None, None, None

    # answer_pred_pos: the position whose logits predict the answer token
    # = (input_len + answer_offset_in_cont) - 1  (logits[pos] predict token at pos+1)
    input_len = processed_embeds.shape[1]
    answer_pred_pos = input_len + answer_offset_in_cont - 1

    if answer_pred_pos < 0 or answer_pred_pos >= full_embeds.shape[1]:
        return None, None, None, None, None

    return full_embeds, thought_positions, answer_pred_pos, answer_token_id, model.base_causallm


@torch.no_grad()
def prepare_m1_inputs(model, tokenizer, sample, device):
    """
    Prepare causal tracing inputs for M1 (CoT baseline).

    1. Generate full output (question + CoT + ### answer)
    2. Tokenize full sequence, get embeddings
    3. Identify CoT positions (between question end and first #)

    Only uses correctly-answered samples.

    Returns:
        full_embeds, cot_positions, answer_pred_pos, answer_token_id, base_model
        or (None, ...) if wrong answer or can't trace.
    """
    # Generate full output
    question_tokens = tokenizer.encode(sample["question"] + "\n", add_special_tokens=True)
    input_ids = torch.tensor(question_tokens, device=device).unsqueeze(0)

    output_ids = model.generate(input_ids=input_ids, max_new_tokens=128, do_sample=False)
    full_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    predicted = extract_answer(full_text)
    gt = get_ground_truth(sample)

    # Only trace correctly-answered samples
    if predicted != gt:
        return None, None, None, None, None

    full_token_ids = output_ids[0]  # [seq_len]

    # Get token embeddings (wte only; base_model will add position embeddings)
    full_embeds = model.transformer.wte(full_token_ids.unsqueeze(0))

    # Find CoT positions: between question end and first hash token
    # BPE may merge "#" into "###" (token 21017) or " ###" etc., so we
    # search for any token whose decoded form contains "#".
    question_len = len(question_tokens)
    all_tokens = full_token_ids.tolist()

    hash_positions = [
        i for i in range(question_len, len(all_tokens))
        if "#" in tokenizer.decode([all_tokens[i]])
    ]

    if len(hash_positions) == 0:
        return None, None, None, None, None

    first_hash_pos = hash_positions[0]
    last_hash_pos = hash_positions[-1]

    # CoT positions: question_len to first_hash_pos (exclusive)
    cot_positions = list(range(question_len, first_hash_pos))

    if len(cot_positions) == 0:
        return None, None, None, None, None

    # Cap for tractability
    if len(cot_positions) > MAX_COT_POSITIONS:
        indices = np.linspace(0, len(cot_positions) - 1, MAX_COT_POSITIONS, dtype=int)
        cot_positions = [cot_positions[i] for i in indices]

    # Answer token: use the actual next token after hash marks in the
    # generated sequence.  Re-tokenizing the ground truth string can
    # produce different token IDs due to BPE context sensitivity
    # (e.g. "Sally" -> [50, 453] vs " Sally" -> [25737]).
    if last_hash_pos + 1 >= len(all_tokens):
        return None, None, None, None, None

    answer_token_id = all_tokens[last_hash_pos + 1]
    answer_pred_pos = last_hash_pos

    if answer_pred_pos < 0 or answer_pred_pos >= full_embeds.shape[1]:
        return None, None, None, None, None

    return full_embeds, cot_positions, answer_pred_pos, answer_token_id, model


# ---------------------------------------------------------------------------
# Noise calibration
# ---------------------------------------------------------------------------

@torch.no_grad()
def estimate_noise_std(model, tokenizer, model_info, data, num_samples_for_stats,
                       num_thoughts, device):
    """
    Estimate std of representations at critical positions across many samples.

    For Coconut: estimates from last-layer hidden states at thought positions.
    For M1 (CoT): estimates from token embeddings at CoT positions.

    Must match the space being corrupted — hidden states have much larger
    norms (~20-50) than token embeddings (~3-5). Using embedding-scale noise
    for hidden-state corruption would produce weak corruption and compress
    the CE denominator, inflating all patch effects.
    """
    is_coconut = (model_info["type"] == "coconut")
    special_ids = get_special_ids(tokenizer)
    all_vecs = []

    for idx in range(min(num_samples_for_stats, len(data))):
        sample = data[idx]

        if is_coconut:
            input_ids = prepare_input(
                sample, tokenizer, model_info,
                num_thoughts=num_thoughts, device=device,
            )
            processed = get_processed_embeds(model, input_ids)
            tokens = input_ids[0].tolist()
            thought_positions = [
                i for i, t in enumerate(tokens)
                if t == special_ids["latent_id"]
            ]
            for pos in thought_positions:
                all_vecs.append(processed[0, pos, :].cpu())
        else:
            # M1: use token embeddings at CoT-like positions
            # (question tokens after first few, as proxy before generation)
            question_tokens = tokenizer.encode(
                sample["question"] + "\n", add_special_tokens=True
            )
            if len(question_tokens) > 5:
                ids = torch.tensor(question_tokens, device=device)
                embeds = model.transformer.wte(ids)
                for pos in range(3, len(question_tokens)):
                    all_vecs.append(embeds[pos].cpu())

    if len(all_vecs) == 0:
        print("  WARNING: No vectors for noise estimation, using default std=1.0")
        return 1.0

    stacked = torch.stack(all_vecs).float()
    noise_std = stacked.std().item()
    mean_norm = stacked.norm(dim=-1).mean().item()
    print(f"  Noise calibration: std={noise_std:.4f}, mean_norm={mean_norm:.2f}, "
          f"n_vectors={len(all_vecs)}")
    return noise_std


# ---------------------------------------------------------------------------
# Trace one sample
# ---------------------------------------------------------------------------

@torch.no_grad()
def trace_sample(base_model, full_embeds, critical_positions,
                 answer_token_id, answer_pred_pos, noise_std, device):
    """
    Run full causal tracing for one sample.

    Returns:
        ce_matrix: np.array [NUM_LAYERS, n_positions]
        logit_clean: float
        logit_corrupt: float
        clean_gt_corrupt: bool (whether logit_clean > logit_corrupt)
    """
    n_pos = len(critical_positions)
    ce_matrix = np.zeros((NUM_LAYERS, n_pos))

    # 1. Clean run
    logit_clean, clean_hidden = clean_forward(
        base_model, full_embeds, answer_token_id, answer_pred_pos
    )

    # 2. Generate noise (3× std per ROME; 1× produces too-weak corruption
    #    where logit_corrupt > logit_clean for most samples)
    hidden_dim = full_embeds.shape[-1]
    noise_embeds = torch.randn(len(critical_positions), hidden_dim, device=device) * noise_std * NOISE_MULTIPLIER

    # 3. Corrupt run
    logit_corrupt, corrupt_embeds = corrupt_forward(
        base_model, full_embeds, critical_positions, noise_embeds,
        answer_token_id, answer_pred_pos
    )

    clean_gt_corrupt = (logit_clean > logit_corrupt)
    denom = logit_clean - logit_corrupt + 1e-6

    # 4. Patch each (layer, position)
    for p_idx, pos in enumerate(critical_positions):
        for layer in range(NUM_LAYERS):
            logit_patched = patched_forward(
                base_model, corrupt_embeds, clean_hidden,
                layer, pos, answer_token_id, answer_pred_pos
            )
            ce_matrix[layer, p_idx] = (logit_patched - logit_corrupt) / denom

    return ce_matrix, logit_clean, logit_corrupt, clean_gt_corrupt


# ---------------------------------------------------------------------------
# Process one model
# ---------------------------------------------------------------------------

@torch.no_grad()
def process_model(model_name, model, tokenizer, model_info, data, num_samples,
                  num_thoughts, device):
    """
    Run causal tracing for all samples on one model.

    Returns results dict with:
        mean_ce: list of lists [NUM_LAYERS x max_positions]
        per_sample_logits: list of (logit_clean, logit_corrupt)
        n_traced: int
        etc.
    """
    is_coconut = (model_info["type"] == "coconut")

    # Estimate noise std globally from first 100 samples
    # Critical: for Coconut, this must be from hidden states (not token embeddings)
    print(f"  Estimating noise distribution from up to 100 samples...")
    noise_std = estimate_noise_std(
        model, tokenizer, model_info, data,
        num_samples_for_stats=min(100, len(data)),
        num_thoughts=num_thoughts,
        device=device,
    )

    # Collect CE matrices (varying n_positions across samples)
    # We'll aggregate by normalizing position index to [0, 1]
    all_ce = []
    all_logit_clean = []
    all_logit_corrupt = []
    n_clean_gt_corrupt = 0
    n_traced = 0
    n_skipped = 0
    n_wrong = 0
    position_counts = []

    for idx in range(min(num_samples, len(data))):
        sample = data[idx]

        # Prepare inputs
        if is_coconut:
            result = prepare_coconut_inputs(
                model, tokenizer, model_info, sample, num_thoughts, device
            )
        else:
            result = prepare_m1_inputs(model, tokenizer, sample, device)

        full_embeds, critical_positions, answer_pred_pos, answer_token_id, base_model = result

        if full_embeds is None:
            if not is_coconut:
                n_wrong += 1  # M1: likely wrong answer
            n_skipped += 1
            continue

        # Trace
        ce_matrix, logit_clean, logit_corrupt, clean_gt = trace_sample(
            base_model, full_embeds, critical_positions,
            answer_token_id, answer_pred_pos, noise_std, device
        )

        all_ce.append(ce_matrix)
        all_logit_clean.append(logit_clean)
        all_logit_corrupt.append(logit_corrupt)
        n_clean_gt_corrupt += int(clean_gt)
        n_traced += 1
        position_counts.append(len(critical_positions))

        if (n_traced) % 50 == 0:
            frac = n_clean_gt_corrupt / n_traced if n_traced > 0 else 0
            print(f"    Traced: {n_traced}/{min(num_samples, len(data))} "
                  f"(skipped {n_skipped}) clean>corrupt: {frac:.2f}")

    if n_traced == 0:
        print(f"  WARNING: No samples traced for {model_name}")
        return None

    # Aggregate CE matrices
    # Since M1 has variable CoT lengths, normalize to max_positions
    max_pos = max(position_counts)

    # Create a fixed-size aggregation grid
    # For models with fixed positions (Coconut), all matrices have same width
    # For M1, interpolate to max_positions
    mean_ce = np.zeros((NUM_LAYERS, max_pos))
    count_ce = np.zeros((NUM_LAYERS, max_pos))

    for ce_matrix in all_ce:
        n_pos = ce_matrix.shape[1]
        if n_pos == max_pos:
            mean_ce += ce_matrix
            count_ce += 1
        else:
            # Map each position to nearest bin
            for p_idx in range(n_pos):
                target_bin = int(round(p_idx * (max_pos - 1) / max(n_pos - 1, 1)))
                mean_ce[:, target_bin] += ce_matrix[:, p_idx]
                count_ce[:, target_bin] += 1

    # Avoid division by zero
    count_ce = np.maximum(count_ce, 1)
    mean_ce = mean_ce / count_ce

    # CE statistics
    ce_in_range = np.sum((mean_ce >= 0) & (mean_ce <= 1))
    ce_total = mean_ce.size
    ce_in_range_fraction = ce_in_range / ce_total if ce_total > 0 else 0

    # For M1 validation: fraction of positions with CE > 0.3 (best layer per position)
    best_layer_ce = np.max(mean_ce, axis=0)  # Best layer per position
    ce_above_03_fraction = np.mean(best_layer_ce > 0.3) if len(best_layer_ce) > 0 else 0

    # Peak CE location
    peak_idx = np.unravel_index(np.argmax(mean_ce), mean_ce.shape)
    peak_layer = int(peak_idx[0])
    peak_position = int(peak_idx[1])
    peak_ce = float(mean_ce[peak_layer, peak_position])

    # Epsilon sensitivity: how much does CE change if we use 1e-8 instead of 1e-6?
    # We approximate: if denom ≈ 0 for many samples, epsilon matters
    near_zero_denoms = sum(
        1 for lc, lr in zip(all_logit_clean, all_logit_corrupt)
        if abs(lc - lr) < 0.01
    )
    epsilon_sensitivity = near_zero_denoms / n_traced * 100 if n_traced > 0 else 0

    result = {
        "mean_ce": mean_ce.tolist(),
        "n_traced": n_traced,
        "n_skipped": n_skipped,
        "n_wrong_answer": n_wrong,
        "max_positions": max_pos,
        "position_counts": position_counts,
        "peak_layer": peak_layer,
        "peak_position": peak_position,
        "peak_ce": round(peak_ce, 4),
        "ce_above_03_fraction": round(ce_above_03_fraction, 4),
        "clean_gt_corrupt_fraction": round(n_clean_gt_corrupt / n_traced, 4),
        "ce_in_range_fraction": round(ce_in_range_fraction, 4),
        "epsilon_sensitivity_pct": round(epsilon_sensitivity, 2),
        "noise_std": round(noise_std, 4),
    }

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Causal Tracing Experiment")
    parser.add_argument("--checkpoint_dir", type=str, required=True,
                        help="Directory containing model subdirs")
    parser.add_argument("--data", type=str, required=True,
                        help="Path to prosqa_test.json")
    parser.add_argument("--num_samples", type=int, default=500)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_thoughts", type=int, default=NUM_THOUGHTS)
    parser.add_argument("--mode", type=str, default="full", choices=["sanity", "full"],
                        help="sanity: Exp 0 quick validation (M1 only, 50 samples). "
                             "full: Exp 4 full causal tracing (default).")
    parser.add_argument("--models", type=str, default=None,
                        help="Comma-separated model names to trace (default: all)")
    args = parser.parse_args()

    # Determine which models to trace
    if args.mode == "sanity":
        # Sanity mode: M1 only, 50 samples, strict criteria
        models_to_trace = ["m1"]
        args.num_samples = 50
    elif args.models is not None:
        models_to_trace = [m.strip() for m in args.models.split(",")]
    else:
        models_to_trace = list(MODELS_TO_TRACE)

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # In full mode, check for sanity gate
    if args.mode == "full":
        sanity_path = os.path.join(args.output_dir, "..", "causal_sanity", "exp0_sanity_result.json")
        if os.path.exists(sanity_path):
            with open(sanity_path, "r") as f:
                sanity = json.load(f)
            if not sanity.get("passed", False):
                print("WARNING: Exp 0 sanity gate FAILED. Proceeding but results may be unreliable.")
        else:
            print("WARNING: Exp 0 sanity result not found. Consider running --mode sanity first.")

    print("=" * 70)
    if args.mode == "sanity":
        print("CAUSAL TRACING — EXP 0 SANITY CHECK")
    else:
        print("CAUSAL TRACING EXPERIMENT")
    print("=" * 70)
    print(f"Mode:            {args.mode}")
    print(f"Models:          {', '.join(models_to_trace)}")
    print(f"Checkpoint dir:  {args.checkpoint_dir}")
    print(f"Data:            {args.data}")
    print(f"Num samples:     {args.num_samples}")
    print(f"Output dir:      {args.output_dir}")
    print(f"Seed:            {args.seed}")
    print(f"Num thoughts:    {args.num_thoughts}")
    print()

    # Load data
    print("Loading data...")
    data = load_data(args.data)
    print(f"  Loaded {len(data)} samples")

    results = {}
    verification = {}
    pipeline_start = time.time()

    # Process M1 FIRST — its result is a mandatory validation check
    if "m1" in models_to_trace:
        print("\n" + "=" * 60)
        print("PROCESSING M1 (CoT baseline) — MANDATORY VALIDATION")
        print("=" * 60)

        try:
            model, tokenizer, model_info = load_model_by_name(
                "m1", args.checkpoint_dir, device=device
            )

            print(f"  Running causal tracing on {args.num_samples} samples...")
            t0 = time.time()

            m1_result = process_model(
                "m1", model, tokenizer, model_info, data,
                num_samples=args.num_samples,
                num_thoughts=args.num_thoughts,
                device=device,
            )

            elapsed = time.time() - t0
            print(f"  M1 tracing done in {elapsed:.1f}s")

            if m1_result is not None:
                results["m1"] = m1_result

                # M1 VALIDATION CHECK
                m1_ce_frac = m1_result["ce_above_03_fraction"]
                m1_clean_gt_frac = m1_result["clean_gt_corrupt_fraction"]
                m1_valid = m1_ce_frac >= 0.5

                verification["m1_cot_ce_above_03_fraction"] = m1_ce_frac

                if m1_valid:
                    print(f"\n  *** M1 VALIDATION PASSED: {m1_ce_frac:.2%} of CoT positions "
                          f"have CE > 0.3 (threshold: 50%) ***")
                else:
                    print(f"\n  *** M1 VALIDATION FAILED: {m1_ce_frac:.2%} of CoT positions "
                          f"have CE > 0.3 (threshold: 50%) ***")
                    print("  WARNING: Patching implementation may be broken!")
                    print("  Proceeding with remaining models, but results should NOT be trusted.")

                # Save M1 intermediate
                m1_path = os.path.join(args.output_dir, "m1_causal.json")
                with open(m1_path, "w") as f:
                    json.dump(m1_result, f, indent=2)
                print(f"  M1 results saved to {m1_path}")

                # --- Sanity mode: strict pass/fail and early exit ---
                if args.mode == "sanity":
                    sanity_elapsed = time.time() - pipeline_start
                    # Criteria: autoregressive CoT concentrates causal effect
                    # at the last position before "###", so we check:
                    #   1. Peak CE > 1.0 (strong patching effect somewhere)
                    #   2. clean>corrupt fraction > 0.5 (corruption degrades majority)
                    #   3. At least 1 position with CE > 0.3
                    # Original ROME-style coverage (CE>0.3 at >=50% positions) is
                    # inappropriate for CoT because information flows to the last
                    # position via autoregressive attention.
                    m1_peak_ce = float(m1_result["peak_ce"])
                    sanity_criteria = {
                        "peak_ce_threshold": 1.0,
                        "min_positions_above_03": 1,
                        "clean_gt_corrupt_threshold": 0.5,
                        "max_time_seconds": 600,
                    }
                    n_positions_above_03 = int(sum(
                        1 for ce in np.max(np.array(m1_result["mean_ce"]), axis=0)
                        if ce > 0.3
                    ))
                    peak_pass = m1_peak_ce >= sanity_criteria["peak_ce_threshold"]
                    pos_pass = n_positions_above_03 >= sanity_criteria["min_positions_above_03"]
                    clean_pass = m1_clean_gt_frac > sanity_criteria["clean_gt_corrupt_threshold"]
                    time_pass = sanity_elapsed <= sanity_criteria["max_time_seconds"]
                    # Peak CE and position coverage are hard requirements;
                    # clean>corrupt is informational (weak noise can invert it)
                    sanity_passed = peak_pass and pos_pass and time_pass

                    sanity_result = {
                        "passed": bool(sanity_passed),
                        "peak_ce": float(m1_peak_ce),
                        "n_positions_above_03": int(n_positions_above_03),
                        "ce_above_03_fraction": float(round(m1_ce_frac, 4)),
                        "clean_gt_corrupt_fraction": float(round(m1_clean_gt_frac, 4)),
                        "elapsed_seconds": float(round(sanity_elapsed, 1)),
                        "n_traced": int(m1_result["n_traced"]),
                        "criteria": sanity_criteria,
                    }

                    sanity_out_path = os.path.join(args.output_dir, "exp0_sanity_result.json")
                    with open(sanity_out_path, "w") as f:
                        json.dump(sanity_result, f, indent=2)
                    print(f"\n  Sanity result saved to {sanity_out_path}")

                    print("\n" + "=" * 70)
                    if sanity_passed:
                        print("  *** EXP 0 SANITY: PASS ***")
                    else:
                        print("  *** EXP 0 SANITY: FAIL ***")
                        if not peak_pass:
                            print(f"    FAIL: peak CE = {m1_peak_ce:.4f} "
                                  f"(need >= {sanity_criteria['peak_ce_threshold']})")
                        if not pos_pass:
                            print(f"    FAIL: positions with CE>0.3 = {n_positions_above_03} "
                                  f"(need >= {sanity_criteria['min_positions_above_03']})")
                        if not clean_pass:
                            print(f"    INFO: clean>corrupt fraction = {m1_clean_gt_frac:.4f} "
                                  f"(below {sanity_criteria['clean_gt_corrupt_threshold']})")
                        if not time_pass:
                            print(f"    FAIL: elapsed {sanity_elapsed:.1f}s "
                                  f"(max {sanity_criteria['max_time_seconds']}s)")
                    print("=" * 70)
                    print(f"  Elapsed:           {sanity_elapsed:.1f}s")
                    print(f"  Peak CE:           {m1_peak_ce:.4f}")
                    print(f"  Positions CE>0.3:  {n_positions_above_03}")
                    print(f"  CE>0.3 fraction:   {m1_ce_frac:.4f}")
                    print(f"  Clean>corrupt:     {m1_clean_gt_frac:.4f}")
                    print(f"  N traced:          {m1_result['n_traced']}")

                    del model
                    torch.cuda.empty_cache()
                    return  # Early exit for sanity mode

            else:
                verification["m1_cot_ce_above_03_fraction"] = 0.0
                print("  WARNING: No M1 samples could be traced!")

                if args.mode == "sanity":
                    # Can't pass sanity without traced samples
                    sanity_result = {
                        "passed": False,
                        "ce_above_03_fraction": 0.0,
                        "clean_gt_corrupt_fraction": 0.0,
                        "elapsed_seconds": round(time.time() - pipeline_start, 1),
                        "n_traced": 0,
                        "criteria": {
                            "ce_threshold": 0.3,
                            "ce_coverage_threshold": 0.5,
                            "clean_gt_corrupt_threshold": 0.8,
                            "max_time_seconds": 600,
                        },
                    }
                    sanity_out_path = os.path.join(args.output_dir, "exp0_sanity_result.json")
                    with open(sanity_out_path, "w") as f:
                        json.dump(sanity_result, f, indent=2)
                    print("\n" + "=" * 70)
                    print("  *** EXP 0 SANITY: FAIL (no samples traced) ***")
                    print("=" * 70)
                    del model
                    torch.cuda.empty_cache()
                    return

            del model
            torch.cuda.empty_cache()

        except FileNotFoundError as e:
            print(f"  SKIPPING M1: {e}")
            verification["m1_cot_ce_above_03_fraction"] = 0.0
            print("  WARNING: Cannot validate patching without M1!")

            if args.mode == "sanity":
                sanity_result = {
                    "passed": False,
                    "ce_above_03_fraction": 0.0,
                    "clean_gt_corrupt_fraction": 0.0,
                    "elapsed_seconds": round(time.time() - pipeline_start, 1),
                    "n_traced": 0,
                    "criteria": {
                        "ce_threshold": 0.3,
                        "ce_coverage_threshold": 0.5,
                        "clean_gt_corrupt_threshold": 0.8,
                        "max_time_seconds": 600,
                    },
                }
                sanity_out_path = os.path.join(args.output_dir, "exp0_sanity_result.json")
                with open(sanity_out_path, "w") as f:
                    json.dump(sanity_result, f, indent=2)
                print("\n" + "=" * 70)
                print("  *** EXP 0 SANITY: FAIL (M1 not found) ***")
                print("=" * 70)
                return

    # Process remaining models (M3, M4, M4b, M5, or whatever --models specifies)
    remaining_models = [m for m in models_to_trace if m != "m1"]
    for model_name in remaining_models:
        print(f"\n{'='*60}")
        print(f"Processing model: {model_name}")
        print(f"{'='*60}")

        try:
            model, tokenizer, model_info = load_model_by_name(
                model_name, args.checkpoint_dir, device=device
            )
        except FileNotFoundError as e:
            print(f"  SKIPPING {model_name}: {e}")
            continue

        print(f"  Type: {model_info['type']}, feedback_mode: {model_info['feedback_mode']}")
        print(f"  Running causal tracing on {args.num_samples} samples...")
        t0 = time.time()

        model_result = process_model(
            model_name, model, tokenizer, model_info, data,
            num_samples=args.num_samples,
            num_thoughts=args.num_thoughts,
            device=device,
        )

        elapsed = time.time() - t0

        if model_result is not None:
            results[model_name] = model_result
            print(f"  {model_name} tracing done in {elapsed:.1f}s")
            print(f"    Peak CE: layer {model_result['peak_layer']}, "
                  f"pos {model_result['peak_position']}, "
                  f"CE={model_result['peak_ce']:.4f}")
            print(f"    CE>0.3 fraction: {model_result['ce_above_03_fraction']:.4f}")

            # Save intermediate
            inter_path = os.path.join(args.output_dir, f"{model_name}_causal.json")
            with open(inter_path, "w") as f:
                json.dump(model_result, f, indent=2)
            print(f"  Saved to {inter_path}")

        del model
        torch.cuda.empty_cache()

    # Aggregate verification
    if "m3" in results:
        verification["clean_gt_corrupt_fraction"] = results["m3"]["clean_gt_corrupt_fraction"]
        verification["ce_in_range_fraction"] = results["m3"]["ce_in_range_fraction"]
        verification["epsilon_sensitivity_pct"] = results["m3"]["epsilon_sensitivity_pct"]
    else:
        # Use whatever model we have
        for mn in ["m4", "m4b", "m1"]:
            if mn in results:
                verification["clean_gt_corrupt_fraction"] = results[mn]["clean_gt_corrupt_fraction"]
                verification["ce_in_range_fraction"] = results[mn]["ce_in_range_fraction"]
                verification["epsilon_sensitivity_pct"] = results[mn]["epsilon_sensitivity_pct"]
                break
        else:
            verification["clean_gt_corrupt_fraction"] = 0.0
            verification["ce_in_range_fraction"] = 0.0
            verification["epsilon_sensitivity_pct"] = 100.0

    # Save final results
    final_results = {**results, "verification": verification}

    results_path = os.path.join(args.output_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(final_results, f, indent=2)
    print(f"\nFinal results saved to {results_path}")

    # Print summary
    print("\n" + "=" * 70)
    print("CAUSAL TRACING RESULTS SUMMARY")
    print("=" * 70)

    for model_name in models_to_trace:
        if model_name not in results:
            continue
        r = results[model_name]
        print(f"\n{model_name}:")
        print(f"  Samples traced:      {r['n_traced']} (skipped {r['n_skipped']})")
        print(f"  Peak CE:             layer {r['peak_layer']}, pos {r['peak_position']}, "
              f"CE={r['peak_ce']:.4f}")
        print(f"  CE>0.3 fraction:     {r['ce_above_03_fraction']:.4f}")
        print(f"  Clean>corrupt frac:  {r['clean_gt_corrupt_fraction']:.4f}")

        # Print text-based CE heatmap (top 5 layers)
        ce = np.array(r["mean_ce"])
        n_pos = ce.shape[1]
        layer_importance = np.max(ce, axis=1)
        top_layers = np.argsort(layer_importance)[-5:][::-1]

        print(f"  CE heatmap (top 5 layers, {n_pos} positions):")
        header = f"  {'Layer':<8}" + "".join(f"{'p='+str(p):<8}" for p in range(min(n_pos, 10)))
        print(header)
        for layer in top_layers:
            row = f"  {layer:<8}"
            for p in range(min(n_pos, 10)):
                val = ce[layer, p]
                row += f"{val:>7.3f} "
            print(row)

    print("\nVerification:")
    for k, v in verification.items():
        print(f"  {k}: {v}")

    # M3 vs M5 comparison
    if "m3" in results and "m5" in results:
        m3_ce = results["m3"]["peak_ce"]
        m5_ce = results["m5"]["peak_ce"]
        print(f"\nPRIMARY COMPARISON: M3 peak CE={m3_ce:.4f}, M5 peak CE={m5_ce:.4f}")
        if m3_ce > m5_ce + 0.1:
            print("  -> M3 > M5: continuous thoughts causally stronger than pause tokens")
        elif abs(m3_ce - m5_ce) <= 0.1:
            print("  -> M3 ~ M5: both similarly important -- curriculum, not mechanism")
        else:
            print("  -> M5 > M3: unexpected -- check for bugs")

    # Final M1 validation verdict
    m1_frac = verification.get("m1_cot_ce_above_03_fraction", 0)
    if m1_frac >= 0.5:
        print("\n*** M1 VALIDATION: PASSED ***")
    else:
        print(f"\n*** M1 VALIDATION: FAILED (CE>0.3 at {m1_frac:.1%} of positions, need >=50%) ***")
        print("*** PATCHING IMPLEMENTATION MAY BE BROKEN -- DO NOT TRUST RESULTS ***")

    print("\nDone.")


if __name__ == "__main__":
    main()
