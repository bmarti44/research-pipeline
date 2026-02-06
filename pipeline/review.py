"""
Research plan review system.

Spawns independent review agents to critically evaluate research plans.
Reviews are "ruthless like a Russian gymnastics coach" but honest, fair, and accurate.
"""

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from enum import Enum


class ReviewSeverity(Enum):
    """Severity of a review finding."""
    CRITICAL = "critical"      # Must fix before proceeding
    MAJOR = "major"            # Should fix, affects validity
    MINOR = "minor"            # Suggestion for improvement
    NOTE = "note"              # Informational


class ReviewCategory(Enum):
    """Category of review finding."""
    HYPOTHESIS = "hypothesis"           # Clarity, testability
    DESIGN = "design"                   # Experimental design issues
    OPERATIONALIZATION = "operationalization"  # How concepts are measured
    STATISTICS = "statistics"           # Analysis plan issues
    CONFOUNDS = "confounds"             # Uncontrolled variables
    CODE = "code"                       # Implementation issues
    REPRODUCIBILITY = "reproducibility" # Can it be replicated
    ETHICS = "ethics"                   # Ethical considerations
    SCOPE = "scope"                     # Overclaiming, limitations


@dataclass
class ReviewFinding:
    """A single finding from a review."""
    id: str
    severity: ReviewSeverity
    category: ReviewCategory
    title: str
    description: str
    location: Optional[str] = None  # File/section where issue is
    recommendation: str = ""
    verified: Optional[bool] = None  # After verification
    verification_notes: str = ""
    implemented: bool = False


@dataclass
class Review:
    """Complete review from one reviewer."""
    reviewer_id: str
    reviewer_perspective: str  # What lens they're reviewing through
    findings: list[ReviewFinding]
    overall_assessment: str
    recommendation: str  # proceed, revise, reject
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict:
        return {
            "reviewer_id": self.reviewer_id,
            "reviewer_perspective": self.reviewer_perspective,
            "findings": [
                {
                    "id": f.id,
                    "severity": f.severity.value,
                    "category": f.category.value,
                    "title": f.title,
                    "description": f.description,
                    "location": f.location,
                    "recommendation": f.recommendation,
                    "verified": f.verified,
                    "verification_notes": f.verification_notes,
                    "implemented": f.implemented,
                }
                for f in self.findings
            ],
            "overall_assessment": self.overall_assessment,
            "recommendation": self.recommendation,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Review":
        findings = [
            ReviewFinding(
                id=f["id"],
                severity=ReviewSeverity(f["severity"]),
                category=ReviewCategory(f["category"]),
                title=f["title"],
                description=f["description"],
                location=f.get("location"),
                recommendation=f.get("recommendation", ""),
                verified=f.get("verified"),
                verification_notes=f.get("verification_notes", ""),
                implemented=f.get("implemented", False),
            )
            for f in data["findings"]
        ]
        return cls(
            reviewer_id=data["reviewer_id"],
            reviewer_perspective=data["reviewer_perspective"],
            findings=findings,
            overall_assessment=data["overall_assessment"],
            recommendation=data["recommendation"],
            timestamp=data.get("timestamp", datetime.now(timezone.utc).isoformat()),
        )


# The 5 reviewer perspectives
REVIEWER_PERSPECTIVES = [
    {
        "id": "methodologist",
        "perspective": "Experimental Design & Methodology",
        "focus": [
            "Is the design appropriate for the hypothesis?",
            "Are there confounds that aren't controlled?",
            "Is the sample size justified?",
            "Are conditions properly operationalized?",
            "Is randomization handled correctly?",
        ],
        "severity_bias": "critical",  # Tends to find critical issues
    },
    {
        "id": "statistician",
        "perspective": "Statistical Analysis & Power",
        "focus": [
            "Is the analysis plan appropriate for the data type?",
            "Are assumptions of statistical tests met?",
            "Is there adequate power to detect the expected effect?",
            "Are multiple comparisons handled?",
            "Is the effect size meaningful and interpretable?",
        ],
        "severity_bias": "major",
    },
    {
        "id": "replicator",
        "perspective": "Reproducibility & Documentation",
        "focus": [
            "Could another researcher replicate this exactly?",
            "Are all parameters specified and locked?",
            "Is the code deterministic?",
            "Are seeds properly set?",
            "Is the environment fully specified?",
        ],
        "severity_bias": "major",
    },
    {
        "id": "skeptic",
        "perspective": "Alternative Explanations & Confounds",
        "focus": [
            "What else could explain these results?",
            "Are there demand characteristics?",
            "Is there selection bias?",
            "Are the conditions actually different in the intended way?",
            "What's the null hypothesis actually testing?",
        ],
        "severity_bias": "critical",
    },
    {
        "id": "implementer",
        "perspective": "Code Quality & Correctness",
        "focus": [
            "Does the code do what it claims?",
            "Are there edge cases not handled?",
            "Is evaluation deterministic and correct?",
            "Are there bugs in the scoring logic?",
            "Does the implementation match the specification?",
        ],
        "severity_bias": "critical",
    },
]


def get_review_prompt(
    perspective: dict,
    research_plan_path: Path,
    study_path: Path,
) -> str:
    """Generate the prompt for a review agent."""

    focus_items = "\n".join(f"- {f}" for f in perspective["focus"])

    return f"""You are a rigorous academic reviewer with expertise in {perspective["perspective"]}.

Your job is to review a research plan with the scrutiny of a Russian gymnastics coach - ruthless but fair, demanding excellence but honest and accurate. You must find real issues, not imagined ones. Be specific and constructive.

## Your Review Focus

{focus_items}

## Review Instructions

1. Read the RESEARCH_PLAN.md carefully
2. Examine the study code (config.yaml, tasks.py, evaluation.py, analysis.py)
3. For EACH issue you find:
   - Assign severity: CRITICAL (must fix), MAJOR (should fix), MINOR (suggestion), NOTE (informational)
   - Assign category: hypothesis, design, operationalization, statistics, confounds, code, reproducibility, ethics, scope
   - Be SPECIFIC about what's wrong and WHERE
   - Provide a concrete RECOMMENDATION for how to fix it

4. If you find NO issues in an area, explicitly state that - don't invent problems
5. Your overall recommendation must be: PROCEED (ready), REVISE (fixable issues), or REJECT (fundamental problems)

## Files to Review

- Research Plan: {research_plan_path}
- Study Directory: {study_path}
  - config.yaml (study configuration)
  - tasks.py (task definitions)
  - evaluation.py (scoring logic)
  - analysis.py (statistical analysis)
  - prompts.py (prompt construction)

## Output Format

Return your review as JSON:
```json
{{
  "reviewer_id": "{perspective['id']}",
  "reviewer_perspective": "{perspective['perspective']}",
  "findings": [
    {{
      "id": "finding_1",
      "severity": "critical|major|minor|note",
      "category": "hypothesis|design|operationalization|statistics|confounds|code|reproducibility|ethics|scope",
      "title": "Brief title",
      "description": "Detailed description of the issue",
      "location": "file.py:line or section name",
      "recommendation": "Specific fix recommendation"
    }}
  ],
  "overall_assessment": "Summary of your review",
  "recommendation": "proceed|revise|reject"
}}
```

Be thorough. Be honest. Be specific. Find real issues, not imagined ones.
"""


def aggregate_reviews(reviews: list[Review]) -> dict:
    """Aggregate findings across all reviews."""
    all_findings = []
    for review in reviews:
        all_findings.extend(review.findings)

    # Count by severity
    by_severity = {
        "critical": len([f for f in all_findings if f.severity == ReviewSeverity.CRITICAL]),
        "major": len([f for f in all_findings if f.severity == ReviewSeverity.MAJOR]),
        "minor": len([f for f in all_findings if f.severity == ReviewSeverity.MINOR]),
        "note": len([f for f in all_findings if f.severity == ReviewSeverity.NOTE]),
    }

    # Count by category
    by_category = {}
    for cat in ReviewCategory:
        count = len([f for f in all_findings if f.category == cat])
        if count > 0:
            by_category[cat.value] = count

    # Overall recommendation (most conservative wins)
    recommendations = [r.recommendation for r in reviews]
    if "reject" in recommendations:
        overall = "reject"
    elif "revise" in recommendations:
        overall = "revise"
    else:
        overall = "proceed"

    return {
        "total_findings": len(all_findings),
        "by_severity": by_severity,
        "by_category": by_category,
        "reviewer_recommendations": {r.reviewer_id: r.recommendation for r in reviews},
        "overall_recommendation": overall,
        "blocking_issues": by_severity["critical"],
    }


def save_reviews(study_path: Path, reviews: list[Review]) -> None:
    """Save all reviews to disk."""
    reviews_path = study_path / "reviews"
    reviews_path.mkdir(exist_ok=True)

    # Save individual reviews
    for review in reviews:
        with open(reviews_path / f"review_{review.reviewer_id}.json", "w") as f:
            json.dump(review.to_dict(), f, indent=2)

    # Save aggregate
    aggregate = aggregate_reviews(reviews)
    with open(reviews_path / "aggregate.json", "w") as f:
        json.dump(aggregate, f, indent=2)

    # Save human-readable summary
    summary = generate_review_summary(reviews, aggregate)
    with open(reviews_path / "REVIEW_SUMMARY.md", "w") as f:
        f.write(summary)


def load_reviews(study_path: Path) -> list[Review]:
    """Load all reviews from disk."""
    reviews_path = study_path / "reviews"
    if not reviews_path.exists():
        return []

    reviews = []
    for review_file in reviews_path.glob("review_*.json"):
        with open(review_file) as f:
            data = json.load(f)
        reviews.append(Review.from_dict(data))

    return reviews


def generate_review_summary(reviews: list[Review], aggregate: dict) -> str:
    """Generate human-readable review summary."""
    summary = f"""# Research Plan Review Summary

Generated: {datetime.now(timezone.utc).isoformat()}

## Overall Recommendation: **{aggregate['overall_recommendation'].upper()}**

## Summary Statistics

- Total Findings: {aggregate['total_findings']}
- Critical Issues: {aggregate['by_severity']['critical']}
- Major Issues: {aggregate['by_severity']['major']}
- Minor Issues: {aggregate['by_severity']['minor']}
- Notes: {aggregate['by_severity']['note']}

## Reviewer Recommendations

| Reviewer | Perspective | Recommendation |
|----------|-------------|----------------|
"""

    for review in reviews:
        summary += f"| {review.reviewer_id} | {review.reviewer_perspective} | {review.recommendation.upper()} |\n"

    summary += "\n## Critical Issues (Must Fix)\n\n"

    critical = [f for r in reviews for f in r.findings if f.severity == ReviewSeverity.CRITICAL]
    if critical:
        for f in critical:
            summary += f"### {f.title}\n\n"
            summary += f"**Category:** {f.category.value}\n"
            summary += f"**Location:** {f.location or 'N/A'}\n\n"
            summary += f"{f.description}\n\n"
            summary += f"**Recommendation:** {f.recommendation}\n\n"
            summary += "---\n\n"
    else:
        summary += "*No critical issues found.*\n\n"

    summary += "## Major Issues (Should Fix)\n\n"

    major = [f for r in reviews for f in r.findings if f.severity == ReviewSeverity.MAJOR]
    if major:
        for f in major:
            summary += f"### {f.title}\n\n"
            summary += f"**Category:** {f.category.value}\n"
            summary += f"**Location:** {f.location or 'N/A'}\n\n"
            summary += f"{f.description}\n\n"
            summary += f"**Recommendation:** {f.recommendation}\n\n"
            summary += "---\n\n"
    else:
        summary += "*No major issues found.*\n\n"

    summary += "## Individual Reviews\n\n"

    for review in reviews:
        summary += f"### {review.reviewer_id}: {review.reviewer_perspective}\n\n"
        summary += f"**Recommendation:** {review.recommendation.upper()}\n\n"
        summary += f"{review.overall_assessment}\n\n"
        summary += f"**Findings:** {len(review.findings)}\n\n"
        summary += "---\n\n"

    return summary


def verify_finding(finding: ReviewFinding, study_path: Path) -> tuple[bool, str]:
    """
    Verify that a review finding is accurate.

    Returns (is_valid, notes).
    """
    # This would contain logic to verify each finding
    # For now, return a placeholder that requires manual verification
    return None, "Requires manual verification"


def mark_finding_verified(
    reviews: list[Review],
    finding_id: str,
    is_valid: bool,
    notes: str,
) -> None:
    """Mark a finding as verified (or not)."""
    for review in reviews:
        for finding in review.findings:
            if finding.id == finding_id:
                finding.verified = is_valid
                finding.verification_notes = notes
                return


def mark_finding_implemented(
    reviews: list[Review],
    finding_id: str,
) -> None:
    """Mark a finding as implemented."""
    for review in reviews:
        for finding in review.findings:
            if finding.id == finding_id:
                finding.implemented = True
                return


def get_unresolved_findings(reviews: list[Review]) -> list[ReviewFinding]:
    """Get findings that are verified but not yet implemented."""
    unresolved = []
    for review in reviews:
        for finding in review.findings:
            if finding.verified is True and not finding.implemented:
                if finding.severity in [ReviewSeverity.CRITICAL, ReviewSeverity.MAJOR]:
                    unresolved.append(finding)
    return unresolved
