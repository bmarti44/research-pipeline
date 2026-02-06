"""
Command-line interface for the research pipeline.

Usage:
    python -m pipeline new <study_name> [--template <template>]
    python -m pipeline run <study_name> [--stage <stage>] [--all] [--full] [--resume]
    python -m pipeline verify <study_name> [--stage <stage>] [--all]
    python -m pipeline prereg <study_name> [--verify]
    python -m pipeline pilot <study_name> [--resume]
    python -m pipeline replicate <original_study> <new_study> [--type direct|conceptual]
    python -m pipeline status [<study_name>]
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

from .runner import (
    run_stage,
    run_all_stages,
    run_full_pipeline,
    run_preregistration,
    run_pilot,
    get_study_status,
    StageStatus,
)
from .verification import verify_stage, verify_all_stages, STAGE_NUMBERS
from .preregistration import (
    create_preregistration,
    verify_preregistration,
    check_preregistration_exists,
)
from .pilot import evaluate_pilot, create_pilot_config, save_pilot_result
from .replication import (
    create_direct_replication,
    create_conceptual_replication,
    generate_replication_report,
)


# Get project root (parent of pipeline/)
PROJECT_ROOT = Path(__file__).parent.parent
STUDIES_DIR = PROJECT_ROOT / "studies"
PAPERS_DIR = PROJECT_ROOT / "papers"
TEMPLATES_DIR = PROJECT_ROOT / "templates"


def find_study_path(study_name: str) -> Path:
    """Find study path in studies/ or papers/*/studies/."""
    # Check studies/ first
    study_path = STUDIES_DIR / study_name
    if study_path.exists():
        return study_path

    # Check papers/*/studies/
    for paper_dir in PAPERS_DIR.iterdir():
        if paper_dir.is_dir():
            study_path = paper_dir / "studies" / study_name
            if study_path.exists():
                return study_path

    return STUDIES_DIR / study_name  # Default location


def cmd_new(args):
    """Create a new study from a template."""
    study_name = args.study_name
    template = args.template or "basic"

    # Determine where to create study
    if args.paper:
        paper_path = PAPERS_DIR / args.paper
        if not paper_path.exists():
            print(f"Error: Paper '{args.paper}' not found at {paper_path}")
            return 1
        study_path = paper_path / "studies" / study_name
    else:
        study_path = STUDIES_DIR / study_name

    template_path = TEMPLATES_DIR / template
    if not template_path.exists():
        # Try templates/study/<template>
        template_path = TEMPLATES_DIR / "study" / template
        if not template_path.exists():
            print(f"Error: Template '{template}' not found")
            available = []
            if TEMPLATES_DIR.exists():
                available.extend([t.name for t in TEMPLATES_DIR.iterdir() if t.is_dir()])
            if (TEMPLATES_DIR / "study").exists():
                available.extend([t.name for t in (TEMPLATES_DIR / "study").iterdir() if t.is_dir()])
            print(f"Available templates: {available}")
            return 1

    if study_path.exists():
        print(f"Error: Study '{study_name}' already exists at {study_path}")
        return 1

    # Copy template to study
    shutil.copytree(template_path, study_path)

    # Rename .template files
    for f in study_path.rglob("*.template"):
        new_name = f.with_suffix("")
        f.rename(new_name)

    # Create stage directories
    for stage, num in STAGE_NUMBERS.items():
        (study_path / "stages" / f"{num}_{stage}").mkdir(parents=True, exist_ok=True)

    # Create outputs directory
    (study_path / "outputs").mkdir(parents=True, exist_ok=True)

    print(f"Created study '{study_name}' at {study_path}")
    print(f"Template: {template}")
    print()
    print("Next steps:")
    print(f"  1. Edit {study_path}/config.yaml")
    print(f"  2. Edit {study_path}/tasks.py")
    print(f"  3. Run: python -m pipeline prereg {study_name}")
    print(f"  4. Run: python -m pipeline run {study_name} --full")

    return 0


def cmd_run(args):
    """Run a study or specific stage."""
    study_name = args.study_name
    study_path = find_study_path(study_name)

    if not study_path.exists():
        print(f"Error: Study '{study_name}' not found")
        return 1

    if args.full:
        # Run full pipeline with all gates
        print(f"Running FULL pipeline for study: {study_name}")
        print(f"  - Includes: preregistration, pilot, adaptive stopping, review gates")
        print()

        results = run_full_pipeline(
            study_path,
            skip_pilot=args.skip_pilot,
            resume=args.resume,
        )

        # Summary
        print()
        all_passed = all(r.status == StageStatus.COMPLETED for r in results.values())
        return 0 if all_passed else 1

    elif args.all:
        print(f"Running all stages for study: {study_name}")
        print()
        results = run_all_stages(study_path, resume=args.resume)

        # Summary
        print()
        print("="*60)
        print("SUMMARY")
        print("="*60)

        all_passed = True
        for stage, result in results.items():
            status = "PASS" if result.status == StageStatus.COMPLETED else "FAIL"
            if result.status != StageStatus.COMPLETED:
                all_passed = False
            print(f"  {stage}: {status}")

        return 0 if all_passed else 1

    elif args.stage:
        print(f"Running stage '{args.stage}' for study: {study_name}")
        print()
        result = run_stage(study_path, args.stage, resume=args.resume)

        if result.status == StageStatus.COMPLETED:
            print(f"\nStage '{args.stage}' completed successfully")
            return 0
        else:
            print(f"\nStage '{args.stage}' failed: {result.error}")
            return 1

    else:
        print("Error: Specify --stage <stage>, --all, or --full")
        return 1


def cmd_verify(args):
    """Verify a study or specific stage."""
    study_name = args.study_name
    study_path = find_study_path(study_name)

    if not study_path.exists():
        print(f"Error: Study '{study_name}' not found")
        return 1

    if args.all:
        print(f"Verifying all stages for study: {study_name}")
        print()
        results = verify_all_stages(study_path)

        all_passed = True
        for stage, result in results.items():
            status = "PASS" if result.passed else "FAIL"
            if not result.passed:
                all_passed = False

            print(f"{stage}:")
            for check in result.checks:
                check_status = "PASS" if check.passed else "FAIL"
                print(f"  [{check_status}] {check.name}: {check.message}")
            print()

        return 0 if all_passed else 1

    elif args.stage:
        print(f"Verifying stage '{args.stage}' for study: {study_name}")
        print()
        result = verify_stage(study_path, args.stage)

        for check in result.checks:
            status = "PASS" if check.passed else "FAIL"
            print(f"  [{status}] {check.name}: {check.message}")

        print()
        if result.passed:
            print(f"Stage '{args.stage}' verification PASSED")
            return 0
        else:
            print(f"Stage '{args.stage}' verification FAILED")
            return 1

    else:
        print("Error: Specify --stage <stage> or --all")
        return 1


def cmd_prereg(args):
    """Create or verify preregistration."""
    study_name = args.study_name
    study_path = find_study_path(study_name)

    if not study_path.exists():
        print(f"Error: Study '{study_name}' not found")
        return 1

    if args.verify:
        # Verify existing preregistration
        if not check_preregistration_exists(study_path):
            print("No preregistration found for this study")
            return 1

        verification = verify_preregistration(study_path)

        if verification.passed:
            print("Preregistration verification PASSED")
            print(f"  Hash: {verification.prereg_hash}")
            return 0
        else:
            print("Preregistration verification FAILED")
            print("  Deviations detected:")
            for dev in verification.deviations:
                print(f"    - {dev}")
            return 1

    else:
        # Create preregistration
        if check_preregistration_exists(study_path):
            print("Warning: Preregistration already exists")
            print("  Use --verify to check compliance")
            return 1

        record = create_preregistration(study_path)

        print("Preregistration created successfully")
        print(f"  Hash: {record.hash}")
        print(f"  Timestamp: {record.timestamp}")
        print(f"  Locked files: {list(record.locked_files.keys())}")
        print()
        print(f"Human-readable version saved to: {study_path}/PREREGISTRATION.md")
        print()
        print("IMPORTANT: Do not modify config.yaml, tasks.py, evaluation.py, or analysis.py")
        print("           after this point. Any changes will be detected at analysis time.")

        return 0


def cmd_pilot(args):
    """Run pilot study."""
    study_name = args.study_name
    study_path = find_study_path(study_name)

    if not study_path.exists():
        print(f"Error: Study '{study_name}' not found")
        return 1

    result = run_pilot(study_path, resume=args.resume)

    if result.status == StageStatus.COMPLETED:
        print("\nPilot study completed successfully")
        print("  You can now run the main study: python -m pipeline run {study_name} --full --skip-pilot")
        return 0
    else:
        print(f"\nPilot study failed: {result.error}")
        return 1


def cmd_replicate(args):
    """Create a replication study."""
    original_name = args.original_study
    new_name = args.new_study
    rep_type = args.type or "direct"

    original_path = find_study_path(original_name)
    if not original_path.exists():
        print(f"Error: Original study '{original_name}' not found")
        return 1

    # Determine new study path
    new_path = original_path.parent / new_name

    try:
        if rep_type == "direct":
            record = create_direct_replication(original_path, new_path, new_seed=args.seed)
            print(f"Direct replication created: {new_path}")
            print(f"  Original hash: {record.original_hash}")
            print(f"  New seed: check config.yaml")

        elif rep_type == "conceptual":
            modifications = {}
            if args.modifications:
                modifications = json.loads(args.modifications)

            record = create_conceptual_replication(original_path, new_path, modifications)
            print(f"Conceptual replication created: {new_path}")
            print(f"  Original hash: {record.original_hash}")
            print(f"  Deviations: {record.deviations}")
            print()
            print("Note: You must edit the study files to implement your modifications")

        else:
            print(f"Error: Unknown replication type '{rep_type}'")
            return 1

        print()
        print("Next steps:")
        print(f"  1. Review/edit {new_path}/config.yaml")
        print(f"  2. Run: python -m pipeline prereg {new_name}")
        print(f"  3. Run: python -m pipeline run {new_name} --full")

        return 0

    except Exception as e:
        print(f"Error creating replication: {e}")
        return 1


def cmd_research(args):
    """Start autonomous research from a hypothesis."""
    hypothesis = args.hypothesis
    paper_name = args.paper or "new_paper"
    study_name = args.study or "study_1"

    print("="*70)
    print("AUTONOMOUS RESEARCH MODE")
    print("="*70)
    print()
    print(f"Hypothesis: {hypothesis}")
    print(f"Paper: {paper_name}")
    print(f"Study: {study_name}")
    print()

    # Create paper directory if needed
    paper_path = PAPERS_DIR / paper_name
    if not paper_path.exists():
        paper_path.mkdir(parents=True)
        (paper_path / "studies").mkdir()
        (paper_path / "manuscript" / "sections").mkdir(parents=True)
        (paper_path / "combined_analysis").mkdir()
        (paper_path / "supplementary" / "data").mkdir(parents=True)

        # Create paper.yaml
        paper_yaml = {
            "paper": {
                "title": f"Research: {hypothesis[:50]}...",
                "status": "in_progress",
            },
            "studies": [{"name": study_name, "role": "Study 1"}],
        }
        import yaml
        with open(paper_path / "paper.yaml", "w") as f:
            yaml.dump(paper_yaml, f)

    # Create study directory
    study_path = paper_path / "studies" / study_name
    if not study_path.exists():
        # Copy template
        template_path = TEMPLATES_DIR / "study" / "tool_calling"
        if template_path.exists():
            import shutil
            shutil.copytree(template_path, study_path)
        else:
            study_path.mkdir(parents=True)

        # Create stage directories
        for i, stage in enumerate(["configure", "generate", "execute", "evaluate", "analyze", "report"], 1):
            (study_path / "stages" / f"{i}_{stage}").mkdir(parents=True, exist_ok=True)
        (study_path / "outputs").mkdir(exist_ok=True)
        (study_path / "reviews").mkdir(exist_ok=True)

    # Initialize research state
    from .orchestrator import ResearchState, save_research_state, get_next_action
    from .interview import get_interview_questions

    state = ResearchState(
        phase="interview",
        hypothesis=hypothesis,
    )
    save_research_state(study_path, state)

    print("Research initialized.")
    print()
    print("NEXT STEPS:")
    print("="*70)
    print()
    print("The agent will now:")
    print("1. Interview you to clarify the hypothesis")
    print("2. Create a detailed RESEARCH_PLAN.md")
    print("3. Conduct 5 independent reviews")
    print("4. Verify and implement review recommendations")
    print("5. Execute the study pipeline")
    print()
    print("All data will be stored in:")
    print(f"  {study_path}/")
    print()
    print("Interview questions to clarify:")
    print("-"*40)

    questions = get_interview_questions()
    for i, q in enumerate(questions[:5], 1):
        print(f"{i}. {q.question}")
        print(f"   Why: {q.why_asking}")
        print()

    print("To continue, answer the interview questions and run:")
    print(f"  python -m pipeline research-continue {study_name}")

    return 0


def cmd_status(args):
    """Show status of studies."""
    if args.study_name:
        # Single study status
        study_path = find_study_path(args.study_name)

        if not study_path.exists():
            print(f"Error: Study '{args.study_name}' not found")
            return 1

        print(f"Study: {args.study_name}")
        print(f"Path: {study_path}")
        print()

        # Check preregistration
        if check_preregistration_exists(study_path):
            verification = verify_preregistration(study_path)
            status = "VALID" if verification.passed else "MODIFIED"
            print(f"Preregistration: {status}")
        else:
            print("Preregistration: NOT CREATED")

        # Check pilot
        pilot_result = study_path / "pilot" / "pilot_result.json"
        if pilot_result.exists():
            with open(pilot_result) as f:
                pr = json.load(f)
            print(f"Pilot: {pr.get('recommendation', 'unknown').upper()}")
        else:
            print("Pilot: NOT RUN")

        print()
        print("Stages:")
        status = get_study_status(study_path)
        for stage, stage_status in status.items():
            print(f"  {stage}: {stage_status.value}")

    else:
        # All studies status
        all_studies = []

        # Studies in studies/
        if STUDIES_DIR.exists():
            all_studies.extend([(d, None) for d in STUDIES_DIR.iterdir() if d.is_dir()])

        # Studies in papers/*/studies/
        if PAPERS_DIR.exists():
            for paper_dir in PAPERS_DIR.iterdir():
                if paper_dir.is_dir():
                    studies_dir = paper_dir / "studies"
                    if studies_dir.exists():
                        all_studies.extend(
                            [(d, paper_dir.name) for d in studies_dir.iterdir() if d.is_dir()]
                        )

        if not all_studies:
            print("No studies found")
            return 0

        print("Studies:")
        print()

        for study_path, paper_name in sorted(all_studies, key=lambda x: x[0].name):
            status = get_study_status(study_path)
            completed = sum(1 for s in status.values() if s == StageStatus.COMPLETED)
            total = len(status)

            prereg = "P" if check_preregistration_exists(study_path) else "-"
            pilot = "I" if (study_path / "pilot" / "pilot_result.json").exists() else "-"

            location = f" [{paper_name}]" if paper_name else ""
            print(f"  {study_path.name}{location}: {completed}/{total} stages [{prereg}{pilot}]")

        print()
        print("Legend: P=preregistered, I=pilot complete")

    return 0


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Research Pipeline for LLM Behavioral Studies",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create and run a study
  python -m pipeline new my_study --template tool_calling
  python -m pipeline prereg my_study           # Lock preregistration
  python -m pipeline run my_study --full       # Full pipeline with gates

  # Individual stages
  python -m pipeline run my_study --stage execute --resume
  python -m pipeline verify my_study --all

  # Pilot testing
  python -m pipeline pilot my_study

  # Replication
  python -m pipeline replicate original_study replication_study --type direct

  # Status
  python -m pipeline status
        """
    )

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # new command
    new_parser = subparsers.add_parser("new", help="Create a new study")
    new_parser.add_argument("study_name", help="Name of the study")
    new_parser.add_argument("--template", "-t", help="Template to use (default: basic)")
    new_parser.add_argument("--paper", "-p", help="Paper to add study to")

    # run command
    run_parser = subparsers.add_parser("run", help="Run a study")
    run_parser.add_argument("study_name", help="Name of the study")
    run_parser.add_argument("--stage", "-s", help="Specific stage to run")
    run_parser.add_argument("--all", "-a", action="store_true", help="Run all stages")
    run_parser.add_argument("--full", "-f", action="store_true", help="Run full pipeline with gates")
    run_parser.add_argument("--resume", "-r", action="store_true", help="Resume from checkpoint")
    run_parser.add_argument("--skip-pilot", action="store_true", help="Skip pilot in full pipeline")

    # verify command
    verify_parser = subparsers.add_parser("verify", help="Verify a study")
    verify_parser.add_argument("study_name", help="Name of the study")
    verify_parser.add_argument("--stage", "-s", help="Specific stage to verify")
    verify_parser.add_argument("--all", "-a", action="store_true", help="Verify all stages")

    # prereg command
    prereg_parser = subparsers.add_parser("prereg", help="Create/verify preregistration")
    prereg_parser.add_argument("study_name", help="Name of the study")
    prereg_parser.add_argument("--verify", "-v", action="store_true", help="Verify existing preregistration")

    # pilot command
    pilot_parser = subparsers.add_parser("pilot", help="Run pilot study")
    pilot_parser.add_argument("study_name", help="Name of the study")
    pilot_parser.add_argument("--resume", "-r", action="store_true", help="Resume from checkpoint")

    # replicate command
    rep_parser = subparsers.add_parser("replicate", help="Create replication study")
    rep_parser.add_argument("original_study", help="Name of original study")
    rep_parser.add_argument("new_study", help="Name of new replication study")
    rep_parser.add_argument("--type", "-t", choices=["direct", "conceptual"], help="Replication type")
    rep_parser.add_argument("--seed", type=int, help="New random seed (for direct replication)")
    rep_parser.add_argument("--modifications", "-m", help="JSON dict of modifications (for conceptual)")

    # status command
    status_parser = subparsers.add_parser("status", help="Show study status")
    status_parser.add_argument("study_name", nargs="?", help="Name of specific study")

    # research command (autonomous mode)
    research_parser = subparsers.add_parser("research", help="Start autonomous research from hypothesis")
    research_parser.add_argument("hypothesis", help="The hypothesis to test")
    research_parser.add_argument("--paper", "-p", help="Paper name (default: new_paper)")
    research_parser.add_argument("--study", "-s", help="Study name (default: study_1)")

    args = parser.parse_args()

    if args.command == "new":
        return cmd_new(args)
    elif args.command == "run":
        return cmd_run(args)
    elif args.command == "verify":
        return cmd_verify(args)
    elif args.command == "prereg":
        return cmd_prereg(args)
    elif args.command == "pilot":
        return cmd_pilot(args)
    elif args.command == "replicate":
        return cmd_replicate(args)
    elif args.command == "status":
        return cmd_status(args)
    elif args.command == "research":
        return cmd_research(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
