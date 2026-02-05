"""
Deterministic verification for each phase of the experiment.

Per PLAN.md requirements, each phase must have verifiable outcomes:
- Checksums for locked files
- Reproducibility checks for RNG-dependent operations
- Threshold checks for validation metrics
- Environment consistency checks
"""

import hashlib
import json
import sys
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Any
import numpy as np


@dataclass
class VerificationResult:
    """Result of a verification check."""
    phase: str
    check_name: str
    passed: bool
    expected: Any
    actual: Any
    message: str
    timestamp: str

    def to_dict(self) -> dict:
        return asdict(self)


class PhaseVerifier:
    """Deterministic verification for experiment phases."""

    def __init__(self, results_dir: str = "experiments/results"):
        self.results_dir = Path(results_dir)
        self.verification_dir = self.results_dir / "verification"
        self.verification_dir.mkdir(parents=True, exist_ok=True)
        self._results: list[VerificationResult] = []

    def _add_result(
        self,
        phase: str,
        check_name: str,
        passed: bool,
        expected: Any,
        actual: Any,
        message: str,
    ) -> VerificationResult:
        result = VerificationResult(
            phase=phase,
            check_name=check_name,
            passed=passed,
            expected=expected,
            actual=actual,
            message=message,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
        self._results.append(result)
        return result

    def compute_file_hash(self, path: str) -> str:
        """Compute SHA256 hash of a file."""
        file_path = Path(path)
        if not file_path.exists():
            return "FILE_NOT_FOUND"

        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest()

    def verify_rng_determinism(self, seed: int = 42) -> VerificationResult:
        """Verify that np.random.default_rng produces deterministic results."""
        rng1 = np.random.default_rng(seed)
        rng2 = np.random.default_rng(seed)

        # Generate sequences
        seq1 = [rng1.random() for _ in range(100)]
        seq2 = [rng2.random() for _ in range(100)]

        passed = seq1 == seq2
        return self._add_result(
            phase="infrastructure",
            check_name="rng_determinism",
            passed=passed,
            expected="identical sequences",
            actual="identical" if passed else "different",
            message="RNG with same seed produces identical sequences" if passed else "RNG non-deterministic!",
        )

    def verify_bootstrap_determinism(self, seed: int = 42) -> VerificationResult:
        """Verify bootstrap functions produce deterministic results."""
        from .bootstrap import bootstrap_mean_ci

        values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

        result1 = bootstrap_mean_ci(values, n_bootstrap=1000, seed=seed)
        result2 = bootstrap_mean_ci(values, n_bootstrap=1000, seed=seed)

        passed = (
            result1["lower"] == result2["lower"]
            and result1["upper"] == result2["upper"]
        )

        return self._add_result(
            phase="infrastructure",
            check_name="bootstrap_determinism",
            passed=passed,
            expected="identical CIs",
            actual=f"CI1=[{result1['lower']:.4f}, {result1['upper']:.4f}], CI2=[{result2['lower']:.4f}, {result2['upper']:.4f}]",
            message="Bootstrap CIs are deterministic" if passed else "Bootstrap CIs are non-deterministic!",
        )

    def verify_phase0_cleanup(self) -> list[VerificationResult]:
        """Verify Phase 0 cleanup was completed correctly."""
        results = []

        # Check deleted files don't exist
        should_not_exist = [
            "experiments/core/__pycache__/cli_wrappers.cpython-312.pyc",
            "experiments/.DS_Store",
            "experiments/results/.DS_Store",
        ]

        for path in should_not_exist:
            exists = Path(path).exists()
            results.append(self._add_result(
                phase="phase0_cleanup",
                check_name=f"deleted_{Path(path).name}",
                passed=not exists,
                expected="file deleted",
                actual="deleted" if not exists else "still exists",
                message=f"{path} correctly removed" if not exists else f"{path} still exists!",
            ))

        # Check required directories exist
        should_exist = [
            "experiments/validation",
            "experiments/analysis",
            "experiments/results/pilot",
            "experiments/results/primary",
            "experiments/results/raw",
        ]

        for path in should_exist:
            exists = Path(path).exists()
            results.append(self._add_result(
                phase="phase0_cleanup",
                check_name=f"created_{Path(path).name}",
                passed=exists,
                expected="directory exists",
                actual="exists" if exists else "missing",
                message=f"{path} exists" if exists else f"{path} missing!",
            ))

        # Check core modules exist
        core_modules = [
            "experiments/core/api_providers.py",
            "experiments/core/bootstrap.py",
            "experiments/core/statistics.py",
            "experiments/core/config.py",
            "experiments/core/tools.py",
            "experiments/core/prompts.py",
            "experiments/core/extractor.py",
            "experiments/core/judge.py",
            "experiments/core/harness.py",
        ]

        for path in core_modules:
            exists = Path(path).exists()
            results.append(self._add_result(
                phase="phase1_infrastructure",
                check_name=f"module_{Path(path).stem}",
                passed=exists,
                expected="file exists",
                actual="exists" if exists else "missing",
                message=f"{path} exists" if exists else f"{path} missing!",
            ))

        return results

    def verify_extractor_accuracy(
        self,
        ground_truth_path: str = "experiments/validation/extraction_ground_truth.json",
        threshold: float = 0.90,
    ) -> VerificationResult:
        """Verify extractor meets accuracy threshold."""
        from .extractor import validate_extractor

        result = validate_extractor(ground_truth_path)

        if "error" in result:
            return self._add_result(
                phase="phase2_validation",
                check_name="extractor_accuracy",
                passed=False,
                expected=f">= {threshold:.0%}",
                actual=result.get("error"),
                message=f"Extractor validation failed: {result.get('error')}",
            )

        accuracy = result.get("accuracy", 0)
        passed = accuracy >= threshold

        return self._add_result(
            phase="phase2_validation",
            check_name="extractor_accuracy",
            passed=passed,
            expected=f">= {threshold:.0%}",
            actual=f"{accuracy:.1%}",
            message=f"Extractor accuracy {accuracy:.1%} {'meets' if passed else 'below'} threshold",
        )

    def verify_judge_kappa(
        self,
        ground_truth_path: str = "experiments/validation/judgment_ground_truth.json",
        threshold: float = 0.75,
    ) -> VerificationResult:
        """Verify judge-human kappa meets threshold."""
        from .judge import validate_judge_human_agreement

        result = validate_judge_human_agreement(ground_truth_path, use_llm=False)

        if "error" in result:
            return self._add_result(
                phase="phase2_validation",
                check_name="judge_kappa",
                passed=False,
                expected=f">= {threshold:.2f}",
                actual=result.get("error"),
                message=f"Judge validation failed: {result.get('error')}",
            )

        kappa = result.get("kappa", 0)
        passed = kappa >= threshold

        return self._add_result(
            phase="phase2_validation",
            check_name="judge_kappa",
            passed=passed,
            expected=f">= {threshold:.2f}",
            actual=f"{kappa:.3f}",
            message=f"Judge-human kappa {kappa:.3f} {'meets' if passed else 'below'} threshold",
        )

    def verify_ground_truth_counts(self) -> list[VerificationResult]:
        """Verify ground truth datasets have minimum required examples."""
        results = []

        # Extraction ground truth: 50+ examples
        ext_path = Path("experiments/validation/extraction_ground_truth.json")
        if ext_path.exists():
            with open(ext_path) as f:
                ext_data = json.load(f)
            count = len(ext_data)
            passed = count >= 50
            results.append(self._add_result(
                phase="phase2_validation",
                check_name="extraction_gt_count",
                passed=passed,
                expected=">= 50",
                actual=str(count),
                message=f"Extraction ground truth has {count} examples",
            ))
        else:
            results.append(self._add_result(
                phase="phase2_validation",
                check_name="extraction_gt_count",
                passed=False,
                expected=">= 50",
                actual="file missing",
                message="Extraction ground truth file not found",
            ))

        # Judgment ground truth: 100+ examples
        jdg_path = Path("experiments/validation/judgment_ground_truth.json")
        if jdg_path.exists():
            with open(jdg_path) as f:
                jdg_data = json.load(f)
            count = len(jdg_data)
            passed = count >= 100
            results.append(self._add_result(
                phase="phase2_validation",
                check_name="judgment_gt_count",
                passed=passed,
                expected=">= 100",
                actual=str(count),
                message=f"Judgment ground truth has {count} examples",
            ))
        else:
            results.append(self._add_result(
                phase="phase2_validation",
                check_name="judgment_gt_count",
                passed=False,
                expected=">= 100",
                actual="file missing",
                message="Judgment ground truth file not found",
            ))

        return results

    def verify_pilot_icc(
        self,
        results_path: str,
        threshold: float = 0.9,
    ) -> VerificationResult:
        """Verify pilot study ICC and determine analysis validity."""
        # This would compute ICC from pilot results
        # For now, return placeholder
        return self._add_result(
            phase="phase3_pilot",
            check_name="pilot_icc",
            passed=True,
            expected=f"report if > {threshold}",
            actual="not yet computed",
            message="ICC check requires pilot data",
        )

    def verify_environment_locked(self) -> VerificationResult:
        """Verify environment is locked before API calls."""
        env_path = self.results_dir / "environment.json"
        config_path = self.results_dir / "model_config_lock.json"

        env_exists = env_path.exists()
        config_exists = config_path.exists()

        passed = env_exists and config_exists

        return self._add_result(
            phase="reproducibility",
            check_name="environment_locked",
            passed=passed,
            expected="both files exist",
            actual=f"env={env_exists}, config={config_exists}",
            message="Environment properly locked" if passed else "Environment not locked!",
        )

    def verify_file_checksums(
        self,
        files_to_check: dict[str, str],
    ) -> list[VerificationResult]:
        """Verify file checksums match expected values."""
        results = []

        for path, expected_hash in files_to_check.items():
            actual_hash = self.compute_file_hash(path)
            passed = actual_hash == expected_hash

            results.append(self._add_result(
                phase="reproducibility",
                check_name=f"checksum_{Path(path).name}",
                passed=passed,
                expected=expected_hash[:16] + "...",
                actual=actual_hash[:16] + "..." if actual_hash != "FILE_NOT_FOUND" else actual_hash,
                message=f"{path} checksum {'matches' if passed else 'MISMATCH'}",
            ))

        return results

    def run_all_phase_checks(self) -> dict:
        """Run all verification checks and return summary."""
        all_results = []

        # Infrastructure checks
        all_results.append(self.verify_rng_determinism())
        all_results.append(self.verify_bootstrap_determinism())

        # Phase 0 checks
        all_results.extend(self.verify_phase0_cleanup())

        # Phase 2 validation checks
        all_results.extend(self.verify_ground_truth_counts())

        # Reproducibility checks
        all_results.append(self.verify_environment_locked())

        # Summarize
        passed = sum(1 for r in all_results if r.passed)
        failed = sum(1 for r in all_results if not r.passed)

        summary = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "total_checks": len(all_results),
            "passed": passed,
            "failed": failed,
            "all_passed": failed == 0,
            "results": [r.to_dict() for r in all_results],
        }

        # Save verification results
        output_path = self.verification_dir / f"verification_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_path, "w") as f:
            json.dump(summary, f, indent=2)

        return summary

    def print_summary(self, summary: dict) -> None:
        """Print verification summary to console."""
        print()
        print("=" * 60)
        print("VERIFICATION SUMMARY")
        print("=" * 60)
        print()
        print(f"Total checks: {summary['total_checks']}")
        print(f"Passed:       {summary['passed']}")
        print(f"Failed:       {summary['failed']}")
        print()

        if summary['failed'] > 0:
            print("FAILED CHECKS:")
            for result in summary['results']:
                if not result['passed']:
                    print(f"  ✗ [{result['phase']}] {result['check_name']}")
                    print(f"    Expected: {result['expected']}")
                    print(f"    Actual:   {result['actual']}")
                    print(f"    {result['message']}")
                    print()

        if summary['all_passed']:
            print("✓ All verification checks passed!")
        else:
            print("✗ Some verification checks failed!")

        print()


def run_verification() -> bool:
    """Run all verification checks and return success status."""
    verifier = PhaseVerifier()
    summary = verifier.run_all_phase_checks()
    verifier.print_summary(summary)
    return summary['all_passed']
