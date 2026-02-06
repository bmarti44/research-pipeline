"""
Replication framework for the research pipeline.

Supports:
1. Direct replication: Exact same study, new sample
2. Conceptual replication: Same hypothesis, different operationalization
3. Extension: Same study + additional conditions
"""

import json
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Literal
import yaml

from .preregistration import get_preregistration_hash, hash_file


@dataclass
class ReplicationRecord:
    """Record of a replication study."""
    original_study: str
    original_hash: str
    replication_type: Literal["direct", "conceptual", "extension"]
    replication_study: str
    deviations: list[str]
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict:
        return {
            "original_study": self.original_study,
            "original_hash": self.original_hash,
            "replication_type": self.replication_type,
            "replication_study": self.replication_study,
            "deviations": self.deviations,
            "timestamp": self.timestamp,
        }


@dataclass
class ReplicationResult:
    """Result of comparing original and replication."""
    original_effect: float
    replication_effect: float
    original_ci: tuple[float, float]
    replication_ci: tuple[float, float]
    original_significant: bool
    replication_significant: bool
    effect_in_same_direction: bool
    effect_in_original_ci: bool
    meta_effect: Optional[float] = None
    meta_ci: Optional[tuple[float, float]] = None
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


def create_direct_replication(
    original_path: Path,
    replication_path: Path,
    new_seed: Optional[int] = None,
) -> ReplicationRecord:
    """
    Create a direct replication of a study.

    Copies all study files but uses a new random seed.
    """
    if replication_path.exists():
        raise ValueError(f"Replication path already exists: {replication_path}")

    # Create replication directory
    replication_path.mkdir(parents=True)

    # Copy study files
    for filename in ["tasks.py", "evaluation.py", "analysis.py", "prompts.py"]:
        src = original_path / filename
        if src.exists():
            shutil.copy(src, replication_path / filename)

    # Load and modify config
    config_path = original_path / "config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Get original hash
    original_hash = get_preregistration_hash(original_path)
    if original_hash is None:
        original_hash = hash_file(config_path)

    # Update for replication
    original_name = config.get("study", {}).get("name", original_path.name)
    config["study"] = config.get("study", {})
    config["study"]["name"] = f"{original_name}_replication"
    config["study"]["replication_of"] = str(original_path)
    config["study"]["replication_type"] = "direct"
    config["study"]["original_hash"] = original_hash

    # New seed
    if new_seed is not None:
        config["trials"] = config.get("trials", {})
        config["trials"]["seed"] = new_seed
    else:
        # Default: original seed + 10000
        original_seed = config.get("trials", {}).get("seed", 42)
        config["trials"] = config.get("trials", {})
        config["trials"]["seed"] = original_seed + 10000

    # Save config
    with open(replication_path / "config.yaml", "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    # Create replication record
    record = ReplicationRecord(
        original_study=str(original_path),
        original_hash=original_hash,
        replication_type="direct",
        replication_study=str(replication_path),
        deviations=["Random seed changed (as expected for direct replication)"],
    )

    # Save record
    with open(replication_path / "replication_record.json", "w") as f:
        json.dump(record.to_dict(), f, indent=2)

    return record


def create_conceptual_replication(
    original_path: Path,
    replication_path: Path,
    modifications: dict,
) -> ReplicationRecord:
    """
    Create a conceptual replication with specified modifications.

    Modifications is a dict describing what changed and why.
    """
    if replication_path.exists():
        raise ValueError(f"Replication path already exists: {replication_path}")

    # Create replication directory
    replication_path.mkdir(parents=True)

    # Load original config
    config_path = original_path / "config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    original_hash = get_preregistration_hash(original_path)
    if original_hash is None:
        original_hash = hash_file(config_path)

    # Note deviations
    deviations = []
    for key, value in modifications.items():
        deviations.append(f"{key}: {value}")

    # Update config
    original_name = config.get("study", {}).get("name", original_path.name)
    config["study"] = config.get("study", {})
    config["study"]["name"] = f"{original_name}_conceptual_rep"
    config["study"]["replication_of"] = str(original_path)
    config["study"]["replication_type"] = "conceptual"
    config["study"]["original_hash"] = original_hash
    config["study"]["modifications"] = modifications

    # Save config (user must provide modified files)
    with open(replication_path / "config.yaml", "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    # Create placeholder files
    for filename in ["tasks.py", "evaluation.py", "analysis.py", "prompts.py"]:
        placeholder = replication_path / filename
        placeholder.write_text(f"# Conceptual replication of {original_path.name}\n"
                               f"# Modify from original: {original_path / filename}\n"
                               f"# Modifications: {modifications.get(filename, 'none specified')}\n")

    record = ReplicationRecord(
        original_study=str(original_path),
        original_hash=original_hash,
        replication_type="conceptual",
        replication_study=str(replication_path),
        deviations=deviations,
    )

    with open(replication_path / "replication_record.json", "w") as f:
        json.dump(record.to_dict(), f, indent=2)

    return record


def compare_replication_results(
    original_path: Path,
    replication_path: Path,
) -> ReplicationResult:
    """
    Compare results from original and replication studies.
    """
    # Load analysis results
    def load_results(study_path: Path) -> dict:
        tests_path = study_path / "stages" / "5_analyze" / "tests.json"
        if not tests_path.exists():
            raise FileNotFoundError(f"No analysis results in {study_path}")

        with open(tests_path) as f:
            tests = json.load(f)

        # Get primary test
        primary = tests[0] if tests else {}
        return {
            "effect": primary.get("effect_size", 0),
            "ci_lower": primary.get("ci_lower", 0),
            "ci_upper": primary.get("ci_upper", 0),
            "p_value": primary.get("p_value", 1.0),
            "significant": primary.get("significant", False),
        }

    original = load_results(original_path)
    replication = load_results(replication_path)

    # Compare
    effect_same_direction = (original["effect"] * replication["effect"]) > 0
    effect_in_ci = (
        original["ci_lower"] <= replication["effect"] <= original["ci_upper"]
    )

    # Simple meta-analysis (inverse variance weighted)
    # This is a rough approximation
    if original["effect"] != 0 and replication["effect"] != 0:
        # Estimate SE from CI (assuming 95% CI)
        original_se = (original["ci_upper"] - original["ci_lower"]) / 3.92
        replication_se = (replication["ci_upper"] - replication["ci_lower"]) / 3.92

        if original_se > 0 and replication_se > 0:
            w1 = 1 / (original_se ** 2)
            w2 = 1 / (replication_se ** 2)

            meta_effect = (w1 * original["effect"] + w2 * replication["effect"]) / (w1 + w2)
            meta_se = 1 / (w1 + w2) ** 0.5
            meta_ci = (meta_effect - 1.96 * meta_se, meta_effect + 1.96 * meta_se)
        else:
            meta_effect = None
            meta_ci = None
    else:
        meta_effect = None
        meta_ci = None

    return ReplicationResult(
        original_effect=original["effect"],
        replication_effect=replication["effect"],
        original_ci=(original["ci_lower"], original["ci_upper"]),
        replication_ci=(replication["ci_lower"], replication["ci_upper"]),
        original_significant=original["significant"],
        replication_significant=replication["significant"],
        effect_in_same_direction=effect_same_direction,
        effect_in_original_ci=effect_in_ci,
        meta_effect=meta_effect,
        meta_ci=meta_ci,
    )


def verify_replication_integrity(replication_path: Path) -> tuple[bool, list[str]]:
    """
    Verify that replication study maintains integrity with original.

    For direct replications, verifies locked files match original.
    """
    record_path = replication_path / "replication_record.json"
    if not record_path.exists():
        return False, ["Not a replication study (no replication_record.json)"]

    with open(record_path) as f:
        record = json.load(f)

    issues = []

    # Check replication type
    rep_type = record.get("replication_type", "unknown")

    if rep_type == "direct":
        # For direct replication, key files should match original
        original_path = Path(record["original_study"])

        if not original_path.exists():
            issues.append(f"Original study not found: {original_path}")
        else:
            for filename in ["tasks.py", "evaluation.py"]:
                original_file = original_path / filename
                rep_file = replication_path / filename

                if original_file.exists() and rep_file.exists():
                    original_hash = hash_file(original_file)
                    rep_hash = hash_file(rep_file)

                    if original_hash != rep_hash:
                        issues.append(f"File differs from original: {filename}")

    elif rep_type == "conceptual":
        # Conceptual replications are expected to differ
        # Just verify record is complete
        if not record.get("deviations"):
            issues.append("Conceptual replication should document deviations")

    return len(issues) == 0, issues


def generate_replication_report(
    original_path: Path,
    replication_path: Path,
    output_path: Optional[Path] = None,
) -> str:
    """
    Generate a comprehensive replication report.
    """
    # Load record
    record_path = replication_path / "replication_record.json"
    with open(record_path) as f:
        record = json.load(f)

    # Compare results
    try:
        comparison = compare_replication_results(original_path, replication_path)
        has_comparison = True
    except FileNotFoundError:
        has_comparison = False
        comparison = None

    # Generate report
    report = f"""# Replication Report

## Study Information

- **Original Study**: {record['original_study']}
- **Original Hash**: {record['original_hash']}
- **Replication Type**: {record['replication_type']}
- **Replication Study**: {record['replication_study']}
- **Timestamp**: {record['timestamp']}

## Documented Deviations

{chr(10).join(f"- {d}" for d in record.get('deviations', [])) or "- None"}

"""

    if has_comparison:
        # Determine replication success
        if comparison.replication_significant and comparison.effect_in_same_direction:
            status = "REPLICATED"
        elif comparison.effect_in_same_direction and comparison.effect_in_original_ci:
            status = "PARTIALLY REPLICATED"
        else:
            status = "NOT REPLICATED"

        report += f"""## Results Comparison

### Replication Status: **{status}**

| Metric | Original | Replication |
|--------|----------|-------------|
| Effect Size | {comparison.original_effect:.3f} | {comparison.replication_effect:.3f} |
| 95% CI | [{comparison.original_ci[0]:.3f}, {comparison.original_ci[1]:.3f}] | [{comparison.replication_ci[0]:.3f}, {comparison.replication_ci[1]:.3f}] |
| Significant | {"Yes" if comparison.original_significant else "No"} | {"Yes" if comparison.replication_significant else "No"} |

### Comparison Criteria

- **Effect in same direction**: {"Yes" if comparison.effect_in_same_direction else "No"}
- **Replication effect within original CI**: {"Yes" if comparison.effect_in_original_ci else "No"}
"""

        if comparison.meta_effect is not None:
            report += f"""
### Meta-Analytic Estimate

- **Combined Effect**: {comparison.meta_effect:.3f}
- **Combined 95% CI**: [{comparison.meta_ci[0]:.3f}, {comparison.meta_ci[1]:.3f}]
"""

    else:
        report += """## Results Comparison

*Analysis not yet complete for both studies.*
"""

    report += """
---
*Generated by Research Pipeline*
"""

    if output_path:
        with open(output_path, "w") as f:
            f.write(report)

    return report
