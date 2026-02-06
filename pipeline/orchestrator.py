"""
Research orchestrator for autonomous hypothesis testing.

Flow:
1. User provides hypothesis
2. Agent interviews user for clarification
3. Agent creates RESEARCH_PLAN.md
4. 5 independent sub-agents review the plan
5. Agent verifies and implements review recommendations
6. Pipeline executes with deterministic verification
"""

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Callable
import yaml

from .interview import (
    InterviewQuestion,
    HypothesisSpec,
    get_interview_questions,
    check_completeness,
    build_hypothesis_spec,
    save_interview,
    format_interview_for_display,
)
from .review import (
    Review,
    ReviewFinding,
    ReviewSeverity,
    REVIEWER_PERSPECTIVES,
    get_review_prompt,
    aggregate_reviews,
    save_reviews,
    load_reviews,
    get_unresolved_findings,
)


@dataclass
class ResearchState:
    """Current state of the research orchestration."""
    phase: str  # interview, planning, review, revision, execution
    hypothesis: str
    spec: Optional[HypothesisSpec] = None
    interview_complete: bool = False
    plan_created: bool = False
    reviews_complete: bool = False
    revisions_complete: bool = False
    execution_started: bool = False
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict:
        return {
            "phase": self.phase,
            "hypothesis": self.hypothesis,
            "spec": self.spec.to_dict() if self.spec else None,
            "interview_complete": self.interview_complete,
            "plan_created": self.plan_created,
            "reviews_complete": self.reviews_complete,
            "revisions_complete": self.revisions_complete,
            "execution_started": self.execution_started,
            "timestamp": self.timestamp,
        }


def generate_research_plan(
    spec: HypothesisSpec,
    study_path: Path,
    paper_name: str,
) -> str:
    """Generate RESEARCH_PLAN.md from hypothesis spec."""

    plan = f"""# Research Plan: {spec.refined_hypothesis}

## Status: DRAFT - Pending Review

Generated: {datetime.now(timezone.utc).isoformat()}

---

## 1. Hypothesis

### Raw Hypothesis
{spec.raw_hypothesis}

### Refined Hypothesis
{spec.refined_hypothesis}

### Expected Direction
{spec.expected_direction}

### Expected Effect Size
{spec.expected_effect_size}

### Theoretical Mechanism
{spec.practical_significance}

---

## 2. Variables

### Independent Variables

| Variable | Levels | Operationalization |
|----------|--------|-------------------|
"""

    for iv in spec.ivs:
        levels = ", ".join(iv.get("levels", []))
        plan += f"| {iv.get('name', 'N/A')} | {levels} | {iv.get('operationalization', 'N/A')} |\n"

    plan += """
### Dependent Variables

| Variable | Measure | Operationalization |
|----------|---------|-------------------|
"""

    for dv in spec.dvs:
        plan += f"| {dv.get('name', 'N/A')} | {dv.get('measure', 'N/A')} | {dv.get('operationalization', 'N/A')} |\n"

    plan += f"""
---

## 3. Design

### Design Type
{spec.design}-subjects design

### Conditions
{chr(10).join(f"- {c}" for c in spec.conditions)}

### Controls
{chr(10).join(f"- {c}" for c in spec.controls) if spec.controls else "- None specified"}

---

## 4. Confounds & Limitations

### Potential Confounds
{chr(10).join(f"- {c}" for c in spec.confounds) if spec.confounds else "- None identified"}

### Known Limitations
{chr(10).join(f"- {l}" for l in spec.limitations) if spec.limitations else "- None specified"}

---

## 5. Population & Scope

### Target Population
{spec.population}

### Generalizability
This study tests the hypothesis specifically in the context of {spec.population}. Results may not generalize to other models or contexts without replication.

---

## 6. Analysis Plan

### Primary Analysis
- Compare {spec.dvs[0].get('name', 'DV')} across conditions using appropriate statistical test
- Expected direction: {spec.expected_direction}
- Minimum effect of interest: {spec.expected_effect_size}

### Statistical Tests
1. Primary: Chi-square test or t-test depending on DV type
2. Effect size: Cohen's d or odds ratio
3. Confidence intervals: 95% bootstrap CI

### Multiple Comparisons
- If multiple DVs: Bonferroni correction
- If multiple conditions: Planned contrasts

### Stopping Rules
- Pilot: Min 20 trials, 95% response rate, 100% deterministic evaluation
- Main: Complete all planned trials unless adaptive stopping triggered

---

## 7. Reproducibility Requirements

### Preregistration
- [ ] All hypothesis and analysis decisions locked before data collection
- [ ] Hash computed and recorded

### Environment Locking
- [ ] Python version locked
- [ ] Package versions locked
- [ ] Model versions locked (no aliases)

### Determinism
- [ ] Random seed specified: 42
- [ ] Evaluation is deterministic (verified in pilot)
- [ ] Analysis is deterministic

---

## 8. Ethical Considerations

- No human subjects involved (LLM testing only)
- Results will be reported honestly regardless of outcome
- Negative results will be published
- All data and code will be made available

---

## 9. Implementation Checklist

### Study Files
- [ ] `config.yaml` - Study configuration with all parameters
- [ ] `tasks.py` - Task definitions with clear specifications
- [ ] `evaluation.py` - Deterministic scoring logic
- [ ] `analysis.py` - Pre-registered statistical tests
- [ ] `prompts.py` - Prompt construction logic

### Verification Gates
- [ ] Preregistration gate (before execute)
- [ ] Pilot gate (before main study)
- [ ] Execution integrity gate (before evaluate)
- [ ] Evaluation determinism gate (before analyze)
- [ ] Preregistration compliance gate (before report)

---

## 10. Review Status

- [ ] Methodologist review
- [ ] Statistician review
- [ ] Replicator review
- [ ] Skeptic review
- [ ] Implementer review

---

*This plan requires 5 independent reviews before execution.*
*All critical and major issues must be resolved.*
"""

    return plan


def save_research_state(study_path: Path, state: ResearchState) -> None:
    """Save current research state."""
    with open(study_path / "research_state.json", "w") as f:
        json.dump(state.to_dict(), f, indent=2)


def load_research_state(study_path: Path) -> Optional[ResearchState]:
    """Load research state if it exists."""
    state_path = study_path / "research_state.json"
    if not state_path.exists():
        return None

    with open(state_path) as f:
        data = json.load(f)

    spec = None
    if data.get("spec"):
        spec = HypothesisSpec(**data["spec"])

    return ResearchState(
        phase=data["phase"],
        hypothesis=data["hypothesis"],
        spec=spec,
        interview_complete=data["interview_complete"],
        plan_created=data["plan_created"],
        reviews_complete=data["reviews_complete"],
        revisions_complete=data["revisions_complete"],
        execution_started=data["execution_started"],
        timestamp=data.get("timestamp", datetime.now(timezone.utc).isoformat()),
    )


def get_next_action(state: ResearchState, study_path: Path) -> dict:
    """Determine what action the orchestrator should take next."""

    if not state.interview_complete:
        return {
            "action": "interview",
            "description": "Continue interviewing user to clarify hypothesis",
            "blocking": True,
        }

    if not state.plan_created:
        return {
            "action": "create_plan",
            "description": "Generate RESEARCH_PLAN.md from hypothesis specification",
            "blocking": False,
        }

    if not state.reviews_complete:
        # Check how many reviews are done
        reviews = load_reviews(study_path)
        done = len(reviews)
        total = len(REVIEWER_PERSPECTIVES)
        return {
            "action": "conduct_reviews",
            "description": f"Conduct reviews ({done}/{total} complete)",
            "blocking": False,
            "reviews_done": done,
            "reviews_total": total,
        }

    # Check if revisions are needed
    reviews = load_reviews(study_path)
    aggregate = aggregate_reviews(reviews)

    if aggregate["blocking_issues"] > 0:
        unresolved = get_unresolved_findings(reviews)
        if unresolved:
            return {
                "action": "implement_revisions",
                "description": f"Implement {len(unresolved)} verified findings",
                "blocking": True,
                "findings": unresolved,
            }
        else:
            return {
                "action": "verify_findings",
                "description": "Verify review findings before implementation",
                "blocking": True,
            }

    if not state.revisions_complete:
        return {
            "action": "mark_revisions_complete",
            "description": "All critical issues resolved - mark revisions complete",
            "blocking": False,
        }

    if not state.execution_started:
        return {
            "action": "start_execution",
            "description": "Begin pipeline execution (preregister → pilot → main study)",
            "blocking": False,
        }

    return {
        "action": "complete",
        "description": "Research orchestration complete",
        "blocking": False,
    }


def format_review_prompt_for_agent(
    perspective: dict,
    research_plan: str,
    study_files: dict,
) -> str:
    """Format the full context for a review agent."""

    prompt = f"""# Review Task: {perspective['perspective']}

You are reviewer "{perspective['id']}" with expertise in {perspective['perspective']}.

## Your Focus Areas
{chr(10).join(f"- {f}" for f in perspective['focus'])}

## Research Plan

```markdown
{research_plan}
```

## Study Files

"""

    for filename, content in study_files.items():
        prompt += f"### {filename}\n\n```python\n{content}\n```\n\n"

    prompt += """
## Review Instructions

1. Review EVERYTHING with extreme rigor - like a Russian gymnastics coach
2. Be HONEST and FAIR - only report real issues, not imagined ones
3. Be ACCURATE - verify your findings against the actual code
4. For each finding, specify:
   - severity: critical (must fix), major (should fix), minor (suggestion), note (FYI)
   - category: hypothesis, design, operationalization, statistics, confounds, code, reproducibility, ethics, scope
   - Exact location (file:line or section)
   - Specific recommendation

5. If something is CORRECT, say so explicitly
6. End with overall recommendation: proceed, revise, or reject

Output your review as structured JSON.
"""

    return prompt


def check_all_reviews_complete(study_path: Path) -> bool:
    """Check if all 5 reviews have been conducted."""
    reviews = load_reviews(study_path)
    return len(reviews) >= len(REVIEWER_PERSPECTIVES)


def check_all_critical_resolved(study_path: Path) -> bool:
    """Check if all critical issues have been resolved."""
    reviews = load_reviews(study_path)
    for review in reviews:
        for finding in review.findings:
            if finding.severity == ReviewSeverity.CRITICAL:
                if not finding.implemented:
                    return False
    return True


def generate_interview_questions_for_user(
    questions: list[InterviewQuestion],
    spec: Optional[HypothesisSpec] = None,
) -> list[dict]:
    """Generate the next set of questions to ask the user."""
    unanswered = [q for q in questions if not q.answer]

    # Group by category for better UX
    by_category = {}
    for q in unanswered[:3]:  # Ask up to 3 at a time
        if q.category not in by_category:
            by_category[q.category] = []
        by_category[q.category].append({
            "id": q.id,
            "question": q.question,
            "why": q.why_asking,
            "required": q.required,
        })

    return list(by_category.values())[0] if by_category else []
