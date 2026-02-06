"""
User interview system for hypothesis clarification.

When given a hypothesis, systematically interviews the user to clarify:
- Independent variables (what's being manipulated)
- Dependent variables (what's being measured)
- Operationalization (how concepts become measurable)
- Expected effect size and direction
- Confounds and controls
- Scope and limitations
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional
import json
from pathlib import Path


@dataclass
class InterviewQuestion:
    """A question to ask the user."""
    id: str
    category: str  # iv, dv, operationalization, effect, confounds, scope
    question: str
    why_asking: str  # Explanation of why this matters
    required: bool = True
    answer: Optional[str] = None
    follow_ups: list[str] = field(default_factory=list)


@dataclass
class HypothesisSpec:
    """Fully specified hypothesis after interview."""
    raw_hypothesis: str
    refined_hypothesis: str

    # Independent variables
    ivs: list[dict]  # [{name, levels, operationalization}]

    # Dependent variables
    dvs: list[dict]  # [{name, measure, scale}]

    # Design
    design: str  # between, within, mixed
    conditions: list[str]

    # Expectations
    expected_direction: str  # positive, negative, null
    expected_effect_size: Optional[str]  # small, medium, large, or specific value
    practical_significance: str  # Why this effect matters

    # Scope
    population: str  # What models/systems this applies to
    limitations: list[str]
    confounds: list[str]
    controls: list[str]

    # Metadata
    interview_complete: bool = False
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict:
        return {
            "raw_hypothesis": self.raw_hypothesis,
            "refined_hypothesis": self.refined_hypothesis,
            "ivs": self.ivs,
            "dvs": self.dvs,
            "design": self.design,
            "conditions": self.conditions,
            "expected_direction": self.expected_direction,
            "expected_effect_size": self.expected_effect_size,
            "practical_significance": self.practical_significance,
            "population": self.population,
            "limitations": self.limitations,
            "confounds": self.confounds,
            "controls": self.controls,
            "interview_complete": self.interview_complete,
            "timestamp": self.timestamp,
        }


# Standard interview questions by category
INTERVIEW_QUESTIONS = [
    # Independent Variables
    InterviewQuestion(
        id="iv_main",
        category="iv",
        question="What is the main thing you're manipulating or comparing? (e.g., 'output format: JSON vs natural language')",
        why_asking="Need to identify the independent variable(s) - what we're changing to see its effect.",
        required=True,
    ),
    InterviewQuestion(
        id="iv_levels",
        category="iv",
        question="What are the specific levels/conditions of this variable? (e.g., 'JSON mode' and 'NL mode')",
        why_asking="Need concrete, implementable conditions for the experiment.",
        required=True,
    ),
    InterviewQuestion(
        id="iv_operationalization",
        category="iv",
        question="How exactly will you implement each condition? What makes them different in practice?",
        why_asking="Operationalization determines if we're actually testing what we think we're testing.",
        required=True,
    ),

    # Dependent Variables
    InterviewQuestion(
        id="dv_main",
        category="dv",
        question="What outcome are you measuring? What counts as 'success' or the effect you're looking for?",
        why_asking="Need to identify the dependent variable(s) - what we're measuring.",
        required=True,
    ),
    InterviewQuestion(
        id="dv_measure",
        category="dv",
        question="How will you measure this? (e.g., binary correct/incorrect, rating scale, count, latency)",
        why_asking="Measurement type affects statistical analysis and interpretation.",
        required=True,
    ),
    InterviewQuestion(
        id="dv_operationalization",
        category="dv",
        question="What specific criteria determine the measurement? (e.g., 'correct if tool name and all required args match')",
        why_asking="Need precise scoring rules that can be implemented deterministically.",
        required=True,
    ),

    # Effect Expectations
    InterviewQuestion(
        id="effect_direction",
        category="effect",
        question="What direction do you expect? (e.g., 'JSON will have LOWER accuracy than NL' or 'no difference expected')",
        why_asking="Pre-specifying direction is required for proper statistical testing and prevents p-hacking.",
        required=True,
    ),
    InterviewQuestion(
        id="effect_size",
        category="effect",
        question="How big an effect do you expect or care about? (e.g., '10 percentage points difference', 'small effect d=0.2')",
        why_asking="Effect size determines sample size needed and whether results are practically meaningful.",
        required=True,
    ),
    InterviewQuestion(
        id="effect_why",
        category="effect",
        question="Why do you expect this effect? What's the theoretical mechanism?",
        why_asking="Understanding the mechanism helps design better tests and interpret unexpected results.",
        required=True,
    ),

    # Confounds and Controls
    InterviewQuestion(
        id="confounds",
        category="confounds",
        question="What else might explain any difference you find? What confounds worry you?",
        why_asking="Identifying confounds early lets us control for them or acknowledge limitations.",
        required=True,
    ),
    InterviewQuestion(
        id="controls",
        category="confounds",
        question="What will you hold constant across conditions? (e.g., same tasks, same model, same temperature)",
        why_asking="Controls ensure differences are due to IV, not other factors.",
        required=True,
    ),

    # Scope
    InterviewQuestion(
        id="population",
        category="scope",
        question="What models/systems should this apply to? (e.g., 'Claude Sonnet specifically' or 'frontier LLMs generally')",
        why_asking="Defines generalizability and whether multi-model testing is needed.",
        required=True,
    ),
    InterviewQuestion(
        id="limitations",
        category="scope",
        question="What are known limitations of this approach? What CAN'T this study tell us?",
        why_asking="Honest limitations prevent overclaiming and guide future work.",
        required=True,
    ),

    # Practical
    InterviewQuestion(
        id="practical_significance",
        category="practical",
        question="Why does this matter? If you find the expected effect, what's the implication?",
        why_asking="Research should have practical or theoretical significance.",
        required=True,
    ),
]


def get_interview_questions() -> list[InterviewQuestion]:
    """Get the standard interview questions."""
    return [InterviewQuestion(
        id=q.id,
        category=q.category,
        question=q.question,
        why_asking=q.why_asking,
        required=q.required,
    ) for q in INTERVIEW_QUESTIONS]


def check_completeness(questions: list[InterviewQuestion]) -> tuple[bool, list[str]]:
    """Check if all required questions have been answered."""
    missing = []
    for q in questions:
        if q.required and not q.answer:
            missing.append(q.id)
    return len(missing) == 0, missing


def build_hypothesis_spec(
    raw_hypothesis: str,
    questions: list[InterviewQuestion],
) -> HypothesisSpec:
    """Build a HypothesisSpec from answered interview questions."""
    answers = {q.id: q.answer for q in questions}

    # Parse IVs
    ivs = [{
        "name": answers.get("iv_main", ""),
        "levels": answers.get("iv_levels", "").split(",") if answers.get("iv_levels") else [],
        "operationalization": answers.get("iv_operationalization", ""),
    }]

    # Parse DVs
    dvs = [{
        "name": answers.get("dv_main", ""),
        "measure": answers.get("dv_measure", ""),
        "operationalization": answers.get("dv_operationalization", ""),
    }]

    # Determine design (default to between-subjects)
    design = "between"  # Could be inferred from answers

    # Build conditions from IV levels
    conditions = [level.strip() for level in ivs[0]["levels"] if level.strip()]

    # Parse confounds and controls
    confounds = [c.strip() for c in answers.get("confounds", "").split(",") if c.strip()]
    controls = [c.strip() for c in answers.get("controls", "").split(",") if c.strip()]
    limitations = [l.strip() for l in answers.get("limitations", "").split(",") if l.strip()]

    # Build refined hypothesis
    iv_name = ivs[0]["name"]
    dv_name = dvs[0]["name"]
    direction = answers.get("effect_direction", "differs")
    refined = f"{iv_name} affects {dv_name}: {direction}"

    complete, _ = check_completeness(questions)

    return HypothesisSpec(
        raw_hypothesis=raw_hypothesis,
        refined_hypothesis=refined,
        ivs=ivs,
        dvs=dvs,
        design=design,
        conditions=conditions,
        expected_direction=answers.get("effect_direction", ""),
        expected_effect_size=answers.get("effect_size", ""),
        practical_significance=answers.get("practical_significance", ""),
        population=answers.get("population", ""),
        limitations=limitations,
        confounds=confounds,
        controls=controls,
        interview_complete=complete,
    )


def save_interview(
    study_path: Path,
    questions: list[InterviewQuestion],
    spec: HypothesisSpec,
) -> None:
    """
    Save interview results as YAML (human-readable collected data).

    This is REQUIRED data collection - must be completed before proceeding.
    """
    import yaml

    # Save as YAML for human readability
    interview_data = {
        "metadata": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "complete": spec.interview_complete,
            "total_questions": len(questions),
            "answered_questions": sum(1 for q in questions if q.answer),
        },
        "hypothesis": {
            "raw": spec.raw_hypothesis,
            "refined": spec.refined_hypothesis,
        },
        "questions_and_answers": [
            {
                "id": q.id,
                "category": q.category,
                "question": q.question,
                "why_asked": q.why_asking,
                "required": q.required,
                "answer": q.answer,
            }
            for q in questions
        ],
        "specification": spec.to_dict(),
    }

    with open(study_path / "interview.yaml", "w") as f:
        yaml.dump(interview_data, f, default_flow_style=False, sort_keys=False)

    # Also save JSON for programmatic access
    with open(study_path / "interview.json", "w") as f:
        json.dump(interview_data, f, indent=2)


def load_interview(study_path: Path) -> tuple[list[InterviewQuestion], HypothesisSpec]:
    """Load previous interview results."""
    import yaml

    # Try YAML first, fall back to JSON
    yaml_path = study_path / "interview.yaml"
    json_path = study_path / "interview.json"

    if yaml_path.exists():
        with open(yaml_path) as f:
            data = yaml.safe_load(f)
    elif json_path.exists():
        with open(json_path) as f:
            data = json.load(f)
    else:
        raise FileNotFoundError(f"No interview data found in {study_path}")

    questions = get_interview_questions()
    for q in questions:
        for saved in data.get("questions_and_answers", data.get("questions", [])):
            if saved["id"] == q.id:
                q.answer = saved.get("answer")

    spec_data = data.get("specification", data.get("spec", {}))
    spec = HypothesisSpec(**spec_data)
    return questions, spec


def verify_interview_complete(study_path: Path) -> tuple[bool, list[str]]:
    """
    Verify that interview was completed (deterministic check).

    This is a REQUIRED verification gate before research can proceed.

    Returns (passed, issues).
    """
    issues = []

    # Check interview file exists
    yaml_path = study_path / "interview.yaml"
    json_path = study_path / "interview.json"

    if not yaml_path.exists() and not json_path.exists():
        issues.append("Interview data not found - interview not conducted")
        return False, issues

    try:
        questions, spec = load_interview(study_path)
    except Exception as e:
        issues.append(f"Failed to load interview data: {e}")
        return False, issues

    # Check all required questions answered
    for q in questions:
        if q.required and not q.answer:
            issues.append(f"Required question not answered: {q.id}")

    # Check spec is complete
    if not spec.interview_complete:
        issues.append("Interview marked as incomplete")

    # Check minimum required fields
    if not spec.refined_hypothesis:
        issues.append("No refined hypothesis")
    if not spec.ivs or not spec.ivs[0].get("name"):
        issues.append("No independent variable specified")
    if not spec.dvs or not spec.dvs[0].get("name"):
        issues.append("No dependent variable specified")
    if not spec.expected_direction:
        issues.append("No expected direction specified")

    return len(issues) == 0, issues


def format_interview_for_display(questions: list[InterviewQuestion]) -> str:
    """Format interview questions and answers for display."""
    output = "# Hypothesis Interview\n\n"

    categories = {
        "iv": "Independent Variables",
        "dv": "Dependent Variables",
        "effect": "Expected Effects",
        "confounds": "Confounds & Controls",
        "scope": "Scope & Limitations",
        "practical": "Practical Significance",
    }

    for cat_id, cat_name in categories.items():
        cat_questions = [q for q in questions if q.category == cat_id]
        if cat_questions:
            output += f"## {cat_name}\n\n"
            for q in cat_questions:
                status = "✓" if q.answer else "○"
                output += f"### {status} {q.question}\n"
                output += f"*Why: {q.why_asking}*\n\n"
                if q.answer:
                    output += f"**Answer:** {q.answer}\n\n"
                else:
                    output += "*Not yet answered*\n\n"

    return output
