"""
Shared utilities for COCONUT experiment scripts.

Handles loading of all 5 model variants (M1-M4b), input preparation,
inference, answer extraction, and hidden state extraction.

Imports from local coconut.py and dataset.py in same directory.
"""

import json
import torch
import torch.nn as nn
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

# Local import from the v9_meta_fork directory
from coconut import Coconut


# ---------------------------------------------------------------------------
# Tokenizer + special token setup
# ---------------------------------------------------------------------------

def setup_tokenizer(model_id="openai-community/gpt2"):
    """Create tokenizer with the 3 special COCONUT tokens added."""
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_tokens("<|start-latent|>")
    tokenizer.add_tokens("<|end-latent|>")
    tokenizer.add_tokens("<|latent|>")
    return tokenizer


def get_special_ids(tokenizer):
    """Return dict of special token ids."""
    return {
        "latent_id": tokenizer.convert_tokens_to_ids("<|latent|>"),
        "start_id": tokenizer.convert_tokens_to_ids("<|start-latent|>"),
        "end_id": tokenizer.convert_tokens_to_ids("<|end-latent|>"),
        "eos_id": tokenizer.eos_token_id,
    }


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(checkpoint_path, device="cuda", feedback_mode=None):
    """
    Load any model type from a state_dict checkpoint.

    Auto-detects whether the checkpoint is a plain GPT-2 (M1/M2) or
    a Coconut wrapper (M3/M4/M4b) by checking if any key starts with
    'base_causallm'.

    For Coconut models, feedback_mode MUST be explicitly provided.
    This prevents silent misidentification of M4/M4b as M3.

    Returns:
        (model, tokenizer, model_info)
        model_info = {
            "type": "cot" | "coconut",
            "feedback_mode": "continuous" | "frozen" | "learned_shared" | None,
        }
    """
    tokenizer = setup_tokenizer()
    special_ids = get_special_ids(tokenizer)

    # Load state dict to inspect keys
    weights = torch.load(checkpoint_path, map_location=device)

    is_coconut = any(k.startswith("base_causallm") for k in weights.keys())

    # Detect checkpoint vocab size from wte.weight shape
    # M1/M2 (plain GPT-2): 50257 (no COCONUT special tokens)
    # M3/M4/M4b (Coconut): 50260 (with <start-latent>, <end-latent>, <latent>)
    wte_key = "base_causallm.transformer.wte.weight" if is_coconut else "transformer.wte.weight"
    ckpt_vocab_size = weights[wte_key].shape[0]

    # Build base GPT-2 model and resize to match checkpoint
    base_model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2")
    if ckpt_vocab_size != base_model.config.vocab_size:
        base_model.resize_token_embeddings(ckpt_vocab_size)
        print(f"[load_model] Resized embeddings: {base_model.config.vocab_size} -> {ckpt_vocab_size}")

    if is_coconut:
        if feedback_mode is None:
            raise ValueError(
                f"Coconut checkpoint detected at {checkpoint_path} but no feedback_mode "
                f"provided. You MUST specify feedback_mode='continuous' (M3), 'frozen' (M4), "
                f"'learned_shared' (M4b), or 'pause_curriculum' (M5). State dicts cannot distinguish these — "
                f"defaulting silently would make M4/M4b behave as M3."
            )
        fm = feedback_mode
        model = Coconut(
            base_model,
            special_ids["latent_id"],
            special_ids["start_id"],
            special_ids["end_id"],
            special_ids["eos_id"],
            feedback_mode=fm,
        )
        result = model.load_state_dict(weights, strict=False)
        print(f"[load_model] Coconut ({fm}) loaded from {checkpoint_path}")
        print(f"  Missing keys: {result.missing_keys}")
        print(f"  Unexpected keys: {result.unexpected_keys}")
        model_info = {"type": "coconut", "feedback_mode": fm}
    else:
        result = base_model.load_state_dict(weights, strict=False)
        print(f"[load_model] Plain GPT-2 loaded from {checkpoint_path}")
        print(f"  Missing keys: {result.missing_keys}")
        print(f"  Unexpected keys: {result.unexpected_keys}")
        model = base_model
        model_info = {"type": "cot", "feedback_mode": None}

    model.eval()
    model.to(device)
    return model, tokenizer, model_info


def load_model_by_name(model_name, checkpoint_dir, device="cuda"):
    """
    Convenience loader that maps model name -> checkpoint path + feedback_mode.

    model_name: "m1" | "m2" | "m3" | "m4" | "m4b" | "m5"
    checkpoint_dir: directory containing prosqa-cot/, prosqa-nocot/, etc.
    """
    import os

    name_to_config = {
        "m1": {"subdir": "prosqa-cot", "feedback_mode": None},
        "m2": {"subdir": "prosqa-nocot", "feedback_mode": None},
        "m3": {"subdir": "prosqa-coconut", "feedback_mode": "continuous"},
        "m4": {"subdir": "prosqa-m4-frozen", "feedback_mode": "frozen"},
        "m4b": {"subdir": "prosqa-m4b-shared", "feedback_mode": "learned_shared"},
        "m5": {"subdir": "prosqa-m5-pause", "feedback_mode": "pause_curriculum"},
    }

    if model_name not in name_to_config:
        raise ValueError(f"Unknown model name '{model_name}'. Expected one of {list(name_to_config.keys())}")

    cfg = name_to_config[model_name]
    ckpt_path = os.path.join(checkpoint_dir, cfg["subdir"], "checkpoint_50")

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    return load_model(ckpt_path, device=device, feedback_mode=cfg["feedback_mode"])


# ---------------------------------------------------------------------------
# Input preparation
# ---------------------------------------------------------------------------

def prepare_input(sample, tokenizer, model_info, num_thoughts=6, device="cuda"):
    """
    Prepare input_ids for a single sample.

    For CoT/direct models (M1/M2): tokenize question only.
    For COCONUT models (M3/M4/M4b): question + start + latent*T + end.

    Args:
        sample: dict with "question", "steps", "answer" keys
        tokenizer: tokenizer with special tokens
        model_info: dict from load_model
        num_thoughts: number of thought tokens (default 6 = max_latent_stage)
        device: torch device

    Returns:
        input_ids: tensor of shape [1, seq_len] on device
    """
    special_ids = get_special_ids(tokenizer)

    question_tokens = tokenizer.encode(sample["question"] + "\n", add_special_tokens=True)

    if model_info["type"] == "coconut":
        # Compute number of thought tokens:
        # min(max_latent_stage, len(steps)) * c_thought
        # With c_thought=1 (default) and max_latent_stage=6 (default)
        n_steps = len(sample.get("steps", []))
        T = min(num_thoughts, n_steps) if n_steps > 0 else num_thoughts
        tokens = (
            question_tokens
            + [special_ids["start_id"]]
            + [special_ids["latent_id"]] * T
            + [special_ids["end_id"]]
        )
    else:
        # M1/M2: just the question
        tokens = question_tokens

    input_ids = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
    return input_ids


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_inference(model, tokenizer, input_ids, model_info, max_new_tokens=64):
    """
    Run generation and return decoded text + extracted answer.

    For Coconut models: uses model.generate() which handles multi-pass thought
    token processing internally.
    For plain models: uses base HF generate().

    Returns:
        (full_text, extracted_answer)
    """
    if model_info["type"] == "coconut":
        attention_mask = torch.ones_like(input_ids, device=input_ids.device)
        output_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
        )
    else:
        output_ids = model.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    full_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    answer = extract_answer(full_text)
    return full_text, answer


@torch.no_grad()
def run_inference_with_embeddings(model, tokenizer, input_ids, model_info, max_new_tokens=64):
    """
    Like run_inference but also returns the inputs_embeds after thought processing.
    Only meaningful for Coconut models.

    Returns:
        (full_text, extracted_answer, inputs_embeds)
        inputs_embeds: tensor [1, seq_len, hidden_dim] or None for plain models
    """
    if model_info["type"] == "coconut":
        attention_mask = torch.ones_like(input_ids, device=input_ids.device)
        output_ids, inputs_embeds = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            output_embedding=True,
        )
        full_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        answer = extract_answer(full_text)
        return full_text, answer, inputs_embeds
    else:
        output_ids = model.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
        full_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        answer = extract_answer(full_text)
        return full_text, answer, None


# ---------------------------------------------------------------------------
# Answer extraction
# ---------------------------------------------------------------------------

def extract_answer(text):
    """
    Extract answer from model output.
    Splits on '#', takes last part, strips commas and whitespace.
    Matches Meta's evaluation: text_output.split("#")[-1].replace(",", "").strip()
    """
    return text.split("#")[-1].replace(",", "").strip()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(path):
    """Load ProsQA JSON dataset."""
    with open(path, "r") as f:
        data = json.load(f)
    return data


# ---------------------------------------------------------------------------
# Hidden state extraction
# ---------------------------------------------------------------------------

@torch.no_grad()
def get_hidden_states(model, tokenizer, input_ids, model_info):
    """
    Forward pass with output_hidden_states=True.

    For Coconut models: runs forward() which does multi-pass thought processing,
    then extracts hidden states from the final forward pass through base_causallm.
    Returns hidden states at all layers for the full sequence.

    For plain models: standard forward with output_hidden_states.

    Returns:
        hidden_states: tensor [num_layers+1, seq_len, hidden_dim]
                       (layer 0 = embedding output, layers 1-12 = transformer layers)
    """
    if model_info["type"] == "coconut":
        # Run the full forward pass through Coconut to get thought-processed embeddings
        attention_mask = torch.ones_like(input_ids, device=input_ids.device)
        labels = input_ids.clone()  # placeholder, not used for loss computation here
        position_ids = torch.arange(
            0, input_ids.shape[1], dtype=torch.long, device=input_ids.device
        ).unsqueeze(0)

        # First get the processed inputs_embeds through the multi-pass loop
        outputs = model.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            position_ids=position_ids,
        )
        inputs_embeds = outputs.inputs_embeds

        # Now run the base model with these processed embeddings to get all hidden states
        base_outputs = model.base_causallm(
            inputs_embeds=inputs_embeds,
            output_hidden_states=True,
        )

        # base_outputs.hidden_states is a tuple of (num_layers+1) tensors
        # each of shape [batch, seq_len, hidden_dim]
        hs_list = base_outputs.hidden_states
        # Stack into [num_layers+1, seq_len, hidden_dim] (remove batch dim)
        hidden_states = torch.stack([h[0] for h in hs_list], dim=0)

    else:
        # Plain GPT-2 model
        outputs = model(
            input_ids=input_ids,
            output_hidden_states=True,
        )
        hs_list = outputs.hidden_states
        hidden_states = torch.stack([h[0] for h in hs_list], dim=0)

    return hidden_states


@torch.no_grad()
def get_thought_hidden_states(model, tokenizer, input_ids, model_info):
    """
    Extract hidden states specifically at thought token positions.

    Returns:
        thought_hidden_states: tensor [num_layers+1, num_thought_tokens, hidden_dim]
        thought_positions: list of int positions of thought tokens in the input
    """
    special_ids = get_special_ids(tokenizer)
    latent_id = special_ids["latent_id"]

    # Find thought token positions
    tokens = input_ids[0].tolist()
    thought_positions = [i for i, t in enumerate(tokens) if t == latent_id]

    if len(thought_positions) == 0:
        return None, []

    # Get full hidden states
    all_hidden = get_hidden_states(model, tokenizer, input_ids, model_info)

    # Extract at thought positions: all_hidden is [layers, seq_len, hidden_dim]
    thought_hidden = all_hidden[:, thought_positions, :]

    return thought_hidden, thought_positions


@torch.no_grad()
def get_processed_embeds(model, input_ids):
    """
    For a Coconut model, run the multi-pass forward and return the final
    inputs_embeds tensor (with thought tokens filled by hidden-state feedback).

    This is the embedding tensor AFTER all thought processing but BEFORE
    the final autoregressive generation.

    Returns:
        inputs_embeds: tensor [1, seq_len, hidden_dim]
    """
    attention_mask = torch.ones_like(input_ids, device=input_ids.device)
    labels = input_ids.clone()
    position_ids = torch.arange(
        0, input_ids.shape[1], dtype=torch.long, device=input_ids.device
    ).unsqueeze(0)

    outputs = model.forward(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        position_ids=position_ids,
    )
    return outputs.inputs_embeds


# ---------------------------------------------------------------------------
# Utility: seed setting
# ---------------------------------------------------------------------------

def set_seed(seed):
    """Set all random seeds for reproducibility."""
    import random
    import os
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Utility: get ground truth answer for a sample
# ---------------------------------------------------------------------------

def get_ground_truth(sample):
    """Extract ground truth answer from a sample, matching Meta's format."""
    return sample["answer"].replace(",", "").strip()


def get_step_types(sample):
    """
    Extract the type/species mentioned at each reasoning step.

    Each step is like "Alex is a yumpus." or "Each yumpus is a rempus."
    The probe target is the last word before the period.

    Returns:
        list of type strings, one per step
    """
    types = []
    for step in sample.get("steps", []):
        # Remove trailing period and whitespace
        step_clean = step.rstrip().rstrip(".")
        # Take the last word
        words = step_clean.split()
        if words:
            types.append(words[-1])
        else:
            types.append("UNKNOWN")
    return types
