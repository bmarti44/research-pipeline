"""
Preregistration system for the research pipeline.

Locks study design, hypotheses, and analysis plans BEFORE data collection.
Provides tamper-evident verification that analysis matches preregistration.
"""

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
import yaml


@dataclass
class PreregistrationRecord:
    """Record of a preregistration."""
    study_name: str
    version: str
    timestamp: str
    hash: str
    hypothesis: str
    primary_dv: str
    analysis_plan: list[str]
    exclusion_criteria: list[str]
    power_analysis: dict
    locked_files: dict[str, str]  # filename -> hash

    def to_dict(self) -> dict:
        return {
            "study_name": self.study_name,
            "version": self.version,
            "timestamp": self.timestamp,
            "hash": self.hash,
            "hypothesis": self.hypothesis,
            "primary_dv": self.primary_dv,
            "analysis_plan": self.analysis_plan,
            "exclusion_criteria": self.exclusion_criteria,
            "power_analysis": self.power_analysis,
            "locked_files": self.locked_files,
        }


@dataclass
class PreregistrationVerification:
    """Result of verifying analysis against preregistration."""
    passed: bool
    prereg_hash: str
    current_hash: str
    deviations: list[str]
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


def hash_content(content: str) -> str:
    """Compute SHA256 hash of content."""
    return f"sha256:{hashlib.sha256(content.encode()).hexdigest()}"


def hash_file(path: Path) -> str:
    """Compute SHA256 hash of a file."""
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)
    return f"sha256:{hasher.hexdigest()}"


def extract_preregistrable_fields(config: dict) -> dict:
    """Extract the fields from config that should be locked."""
    study = config.get("study", {})
    analysis = config.get("analysis", {})
    trials = config.get("trials", {})

    return {
        "hypothesis": study.get("hypothesis", ""),
        "primary_dv": study.get("primary_dv", ""),
        "secondary_dvs": study.get("secondary_dvs", []),
        "analysis_plan": analysis.get("primary_tests", []),
        "secondary_analyses": analysis.get("secondary_tests", []),
        "exclusion_criteria": study.get("exclusion_criteria", []),
        "power": {
            "effect_size": analysis.get("power", {}).get("effect_size"),
            "alpha": analysis.get("power", {}).get("alpha", 0.05),
            "power_target": analysis.get("power", {}).get("power", 0.80),
            "sample_size": trials.get("total", None),
        },
        "conditions": [c.get("name") for c in config.get("conditions", [])],
        "randomization_seed": trials.get("seed"),
    }


def create_preregistration(study_path: Path) -> PreregistrationRecord:
    """
    Create a preregistration from the current study configuration.

    This MUST be called BEFORE running the execute stage.
    """
    # Load config
    config_path = study_path / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"No config.yaml found in {study_path}")

    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Extract preregistrable fields
    fields = extract_preregistrable_fields(config)

    # Hash key files that define the study
    locked_files = {}
    for filename in ["config.yaml", "tasks.py", "evaluation.py", "analysis.py"]:
        file_path = study_path / filename
        if file_path.exists():
            locked_files[filename] = hash_file(file_path)

    # Create overall hash
    content_to_hash = json.dumps({
        "fields": fields,
        "locked_files": locked_files,
    }, sort_keys=True)
    overall_hash = hash_content(content_to_hash)

    # Create record
    record = PreregistrationRecord(
        study_name=config.get("study", {}).get("name", study_path.name),
        version="1.0.0",
        timestamp=datetime.now(timezone.utc).isoformat(),
        hash=overall_hash,
        hypothesis=fields["hypothesis"],
        primary_dv=fields["primary_dv"],
        analysis_plan=fields["analysis_plan"],
        exclusion_criteria=fields["exclusion_criteria"],
        power_analysis=fields["power"],
        locked_files=locked_files,
    )

    # Save preregistration
    prereg_path = study_path / "preregistration.json"
    with open(prereg_path, "w") as f:
        json.dump(record.to_dict(), f, indent=2)

    # Also create human-readable version
    human_readable = f"""# Preregistration: {record.study_name}

## Locked At
{record.timestamp}

## Hash
{record.hash}

## Hypothesis
{record.hypothesis}

## Primary Dependent Variable
{record.primary_dv}

## Analysis Plan
{chr(10).join(f"- {a}" for a in record.analysis_plan) if record.analysis_plan else "- Not specified"}

## Exclusion Criteria
{chr(10).join(f"- {e}" for e in record.exclusion_criteria) if record.exclusion_criteria else "- None specified"}

## Power Analysis
- Target effect size: {record.power_analysis.get('effect_size', 'N/A')}
- Alpha: {record.power_analysis.get('alpha', 0.05)}
- Power: {record.power_analysis.get('power_target', 0.80)}
- Sample size: {record.power_analysis.get('sample_size', 'N/A')}

## Locked Files
{chr(10).join(f"- {f}: {h}" for f, h in record.locked_files.items())}

---
*This document was auto-generated and should not be modified after creation.*
"""

    with open(study_path / "PREREGISTRATION.md", "w") as f:
        f.write(human_readable)

    return record


def verify_preregistration(study_path: Path) -> PreregistrationVerification:
    """
    Verify that current study files match the preregistration.

    Call this BEFORE running analysis to ensure no post-hoc changes.
    """
    prereg_path = study_path / "preregistration.json"
    if not prereg_path.exists():
        return PreregistrationVerification(
            passed=False,
            prereg_hash="",
            current_hash="",
            deviations=["No preregistration.json found - study was not preregistered"],
        )

    with open(prereg_path) as f:
        prereg = json.load(f)

    deviations = []

    # Check locked files
    for filename, expected_hash in prereg.get("locked_files", {}).items():
        file_path = study_path / filename
        if not file_path.exists():
            deviations.append(f"Locked file missing: {filename}")
        else:
            current_hash = hash_file(file_path)
            if current_hash != expected_hash:
                deviations.append(
                    f"File modified after preregistration: {filename} "
                    f"(expected {expected_hash[:20]}..., got {current_hash[:20]}...)"
                )

    # Recompute overall hash
    config_path = study_path / "config.yaml"
    if config_path.exists():
        with open(config_path) as f:
            config = yaml.safe_load(f)

        fields = extract_preregistrable_fields(config)

        current_locked = {}
        for filename in prereg.get("locked_files", {}).keys():
            file_path = study_path / filename
            if file_path.exists():
                current_locked[filename] = hash_file(file_path)

        content_to_hash = json.dumps({
            "fields": fields,
            "locked_files": current_locked,
        }, sort_keys=True)
        current_hash = hash_content(content_to_hash)
    else:
        current_hash = ""
        deviations.append("config.yaml not found")

    passed = len(deviations) == 0 and current_hash == prereg.get("hash", "")

    return PreregistrationVerification(
        passed=passed,
        prereg_hash=prereg.get("hash", ""),
        current_hash=current_hash,
        deviations=deviations,
    )


def check_preregistration_exists(study_path: Path) -> bool:
    """Check if study has been preregistered."""
    return (study_path / "preregistration.json").exists()


def get_preregistration_hash(study_path: Path) -> Optional[str]:
    """Get the preregistration hash if it exists."""
    prereg_path = study_path / "preregistration.json"
    if not prereg_path.exists():
        return None

    with open(prereg_path) as f:
        prereg = json.load(f)

    return prereg.get("hash")
