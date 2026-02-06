"""Model implementations for LAHR architecture study.

CANONICAL MODEL: lahr_v4.py - use this for all experiments.
"""

from .layers import RMSNorm, MambaSSM, SelectiveAttention, AdaptiveDepthBlock, FeedForward
from .ssh import SSH, SSHConfig, create_ssh_small, create_ssh_medium
from .baseline import BaselineTransformer, TransformerConfig, create_transformer_small, create_transformer_medium

# LAHR v4 - Canonical implementation (use this)
from .lahr_v4 import (
    LAHRv4,
    LAHRConfig,
    create_lahr_tiny,
    create_lahr_small,
    create_lahr_medium,
    create_baseline as create_lahr_baseline,
    # Full 2^3 factorial ablation factories
    create_ablation,
    create_ablation_full,
    create_ablation_no_mod,
    create_ablation_no_latent,
    create_ablation_no_memory,
    create_ablation_mod_only,
    create_ablation_latent_only,
    create_ablation_memory_only,
    create_ablation_baseline,
    ABLATION_FACTORY,
)

__all__ = [
    # Shared layers
    "RMSNorm",
    "MambaSSM",
    "SelectiveAttention",
    "AdaptiveDepthBlock",
    "FeedForward",
    # SSH (legacy)
    "SSH",
    "SSHConfig",
    "create_ssh_small",
    "create_ssh_medium",
    # Baseline transformer
    "BaselineTransformer",
    "TransformerConfig",
    "create_transformer_small",
    "create_transformer_medium",
    # LAHR v4 (CANONICAL)
    "LAHRv4",
    "LAHRConfig",
    "create_lahr_tiny",
    "create_lahr_small",
    "create_lahr_medium",
    "create_lahr_baseline",
    # Ablation factories
    "create_ablation",
    "create_ablation_full",
    "create_ablation_no_mod",
    "create_ablation_no_latent",
    "create_ablation_no_memory",
    "create_ablation_mod_only",
    "create_ablation_latent_only",
    "create_ablation_memory_only",
    "create_ablation_baseline",
    "ABLATION_FACTORY",
]
