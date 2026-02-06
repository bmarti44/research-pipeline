"""
Manuscript generation for the research pipeline.

Provides functions for:
- Generating manuscript sections from study results
- Creating tables and figures
- Formatting statistical results
- Assembling complete manuscripts
"""

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional
import textwrap

from .utils import now_iso, json_load, read_yaml


# =============================================================================
# Manuscript Components
# =============================================================================

@dataclass
class ManuscriptSection:
    """A section of the manuscript."""
    name: str
    title: str
    content: str
    order: int = 0
    subsections: list["ManuscriptSection"] = field(default_factory=list)

    def to_markdown(self, level: int = 1) -> str:
        """Convert to markdown format."""
        header = "#" * level
        lines = [f"{header} {self.title}", "", self.content, ""]

        for subsection in sorted(self.subsections, key=lambda s: s.order):
            lines.append(subsection.to_markdown(level + 1))

        return "\n".join(lines)


@dataclass
class Table:
    """A table for the manuscript."""
    name: str
    caption: str
    headers: list[str]
    rows: list[list[str]]
    notes: Optional[str] = None

    def to_markdown(self) -> str:
        """Convert to markdown table format."""
        lines = [f"**{self.caption}**", ""]

        # Header row
        lines.append("| " + " | ".join(self.headers) + " |")
        lines.append("| " + " | ".join(["---"] * len(self.headers)) + " |")

        # Data rows
        for row in self.rows:
            lines.append("| " + " | ".join(str(cell) for cell in row) + " |")

        if self.notes:
            lines.append("")
            lines.append(f"*Note: {self.notes}*")

        return "\n".join(lines)

    def to_latex(self) -> str:
        """Convert to LaTeX table format."""
        col_spec = "l" + "c" * (len(self.headers) - 1)
        lines = [
            "\\begin{table}[htbp]",
            "\\centering",
            f"\\caption{{{self.caption}}}",
            f"\\begin{{tabular}}{{{col_spec}}}",
            "\\toprule",
            " & ".join(self.headers) + " \\\\",
            "\\midrule",
        ]

        for row in self.rows:
            lines.append(" & ".join(str(cell) for cell in row) + " \\\\")

        lines.extend([
            "\\bottomrule",
            "\\end{tabular}",
        ])

        if self.notes:
            lines.append(f"\\caption*{{Note: {self.notes}}}")

        lines.append("\\end{table}")

        return "\n".join(lines)


@dataclass
class Figure:
    """A figure reference for the manuscript."""
    name: str
    caption: str
    path: Path
    notes: Optional[str] = None

    def to_markdown(self) -> str:
        """Convert to markdown figure format."""
        lines = [
            f"![{self.caption}]({self.path})",
            "",
            f"*Figure: {self.caption}*",
        ]
        if self.notes:
            lines.append(f"*Note: {self.notes}*")
        return "\n".join(lines)


@dataclass
class Manuscript:
    """Complete manuscript."""
    title: str
    authors: list[dict]
    abstract: str
    sections: list[ManuscriptSection]
    tables: list[Table] = field(default_factory=list)
    figures: list[Figure] = field(default_factory=list)
    references: list[dict] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    def to_markdown(self) -> str:
        """Convert complete manuscript to markdown."""
        lines = [
            f"# {self.title}",
            "",
            self._format_authors(),
            "",
            "## Abstract",
            "",
            self.abstract,
            "",
        ]

        # Sections
        for section in sorted(self.sections, key=lambda s: s.order):
            lines.append(section.to_markdown(level=2))

        # Tables
        if self.tables:
            lines.append("## Tables")
            lines.append("")
            for table in self.tables:
                lines.append(table.to_markdown())
                lines.append("")

        # Figures
        if self.figures:
            lines.append("## Figures")
            lines.append("")
            for figure in self.figures:
                lines.append(figure.to_markdown())
                lines.append("")

        # References
        if self.references:
            lines.append("## References")
            lines.append("")
            for i, ref in enumerate(self.references, 1):
                lines.append(f"{i}. {self._format_reference(ref)}")
            lines.append("")

        return "\n".join(lines)

    def _format_authors(self) -> str:
        """Format author list."""
        author_strs = []
        for author in self.authors:
            name = author.get("name", "Unknown")
            affiliation = author.get("affiliation", "")
            if affiliation:
                author_strs.append(f"{name} ({affiliation})")
            else:
                author_strs.append(name)
        return ", ".join(author_strs)

    def _format_reference(self, ref: dict) -> str:
        """Format a single reference."""
        authors = ref.get("authors", "Unknown")
        year = ref.get("year", "n.d.")
        title = ref.get("title", "Untitled")
        journal = ref.get("journal", "")
        doi = ref.get("doi", "")

        parts = [f"{authors} ({year}). {title}."]
        if journal:
            parts.append(f" *{journal}*.")
        if doi:
            parts.append(f" https://doi.org/{doi}")

        return "".join(parts)


# =============================================================================
# Section Generators
# =============================================================================

def generate_introduction(
    hypothesis: str,
    research_questions: list[dict],
    background: Optional[str] = None,
) -> ManuscriptSection:
    """
    Generate the Introduction section.

    Args:
        hypothesis: Study hypothesis
        research_questions: List of research question dicts
        background: Optional background text

    Returns:
        ManuscriptSection for Introduction
    """
    content_parts = []

    if background:
        content_parts.append(background)
        content_parts.append("")

    content_parts.append("### Hypothesis")
    content_parts.append("")
    content_parts.append(hypothesis)
    content_parts.append("")

    if research_questions:
        content_parts.append("### Research Questions")
        content_parts.append("")
        for rq in research_questions:
            rq_id = rq.get("id", "")
            question = rq.get("question", "")
            content_parts.append(f"- **{rq_id}**: {question}")
        content_parts.append("")

    return ManuscriptSection(
        name="introduction",
        title="Introduction",
        content="\n".join(content_parts),
        order=1,
    )


def generate_methods(
    config: dict,
    n_trials: int,
    n_tasks: int,
) -> ManuscriptSection:
    """
    Generate the Methods section.

    Args:
        config: Study configuration
        n_trials: Total number of trials
        n_tasks: Number of unique tasks

    Returns:
        ManuscriptSection for Methods
    """
    conditions = config.get("conditions", [])
    models = config.get("models", [])
    repetitions = config.get("trials", {}).get("repetitions", 1)

    content_parts = []

    # Design
    content_parts.append("### Design")
    content_parts.append("")
    content_parts.append(
        f"This study used a within-subjects design with {len(conditions)} conditions:"
    )
    for cond in conditions:
        name = cond.get("name", "unnamed")
        desc = cond.get("description", "")
        content_parts.append(f"- **{name}**: {desc}")
    content_parts.append("")

    # Materials
    content_parts.append("### Materials")
    content_parts.append("")
    content_parts.append(f"We used {n_tasks} tasks across the following categories:")
    # Categories would be extracted from tasks
    content_parts.append("")

    # Procedure
    content_parts.append("### Procedure")
    content_parts.append("")
    content_parts.append(
        f"Each task was presented {repetitions} times per condition, "
        f"yielding {n_trials} total trials. "
        f"Trials were randomized within participants."
    )
    content_parts.append("")

    # Model
    if models:
        content_parts.append("### Model")
        content_parts.append("")
        for model in models:
            model_id = model.get("model_id", model.get("alias", "unknown"))
            provider = model.get("provider", "unknown")
            temp = model.get("temperature", "default")
            content_parts.append(
                f"We used {model_id} from {provider} with temperature={temp}."
            )
        content_parts.append("")

    return ManuscriptSection(
        name="methods",
        title="Methods",
        content="\n".join(content_parts),
        order=2,
    )


def generate_results(
    aggregates: dict,
    tests: list[dict],
    alpha: float = 0.05,
) -> ManuscriptSection:
    """
    Generate the Results section.

    Args:
        aggregates: Aggregated results by condition
        tests: Statistical test results
        alpha: Significance level

    Returns:
        ManuscriptSection for Results
    """
    content_parts = []

    # Descriptive statistics
    content_parts.append("### Descriptive Statistics")
    content_parts.append("")

    for condition, stats in aggregates.items():
        n = stats.get("n", 0)
        rate = stats.get("rate", 0)
        ci_lower = stats.get("ci_lower", 0)
        ci_upper = stats.get("ci_upper", 1)
        content_parts.append(
            f"**{condition}**: {rate:.1%} correct (95% CI [{ci_lower:.1%}, {ci_upper:.1%}], n={n})"
        )
    content_parts.append("")

    # Statistical tests
    content_parts.append("### Inferential Statistics")
    content_parts.append("")

    for test in tests:
        test_name = test.get("name", "Test")
        stat = test.get("statistic")
        p = test.get("p_value")
        significant = test.get("significant", False)
        effect = test.get("effect_size")

        # Format p-value
        if p is not None:
            if p < 0.001:
                p_str = "p < .001"
            else:
                p_str = f"p = {p:.3f}"
        else:
            p_str = "p = N/A"

        # Format statistic
        if stat is not None:
            stat_str = f"{stat:.2f}"
        else:
            stat_str = "N/A"

        content_parts.append(f"**{test_name}**: statistic = {stat_str}, {p_str}")

        if significant:
            content_parts.append(f"  - Result is statistically significant at α = {alpha}")
        else:
            content_parts.append(f"  - Result is not statistically significant at α = {alpha}")

        if effect is not None:
            content_parts.append(f"  - Effect size: {effect:.3f}")

        content_parts.append("")

    return ManuscriptSection(
        name="results",
        title="Results",
        content="\n".join(content_parts),
        order=3,
    )


def generate_discussion(
    hypothesis: str,
    significant: bool,
    effect_size: Optional[float] = None,
) -> ManuscriptSection:
    """
    Generate a basic Discussion section template.

    Args:
        hypothesis: Study hypothesis
        significant: Whether main test was significant
        effect_size: Effect size if available

    Returns:
        ManuscriptSection for Discussion
    """
    content_parts = []

    if significant:
        content_parts.append(
            "The results support the hypothesis that " +
            hypothesis.lower().rstrip(".") + "."
        )
    else:
        content_parts.append(
            "The results do not provide support for the hypothesis that " +
            hypothesis.lower().rstrip(".") + "."
        )

    content_parts.append("")
    content_parts.append("### Limitations")
    content_parts.append("")
    content_parts.append("[Limitations to be added]")
    content_parts.append("")
    content_parts.append("### Future Directions")
    content_parts.append("")
    content_parts.append("[Future directions to be added]")

    return ManuscriptSection(
        name="discussion",
        title="Discussion",
        content="\n".join(content_parts),
        order=4,
    )


# =============================================================================
# Table Generators
# =============================================================================

def generate_results_table(
    aggregates: dict,
    caption: str = "Results by Condition",
) -> Table:
    """
    Generate a results table from aggregated data.

    Args:
        aggregates: Aggregated results by condition
        caption: Table caption

    Returns:
        Table object
    """
    headers = ["Condition", "N", "Correct", "Rate", "95% CI"]
    rows = []

    for condition, stats in aggregates.items():
        n = stats.get("n", 0)
        correct = stats.get("correct", stats.get("successes", 0))
        rate = stats.get("rate", 0)
        ci_lower = stats.get("ci_lower", 0)
        ci_upper = stats.get("ci_upper", 1)

        ci_str = f"[{ci_lower:.3f}, {ci_upper:.3f}]"
        rows.append([condition, str(n), str(correct), f"{rate:.3f}", ci_str])

    return Table(
        name="results_table",
        caption=caption,
        headers=headers,
        rows=rows,
    )


def generate_test_table(
    tests: list[dict],
    caption: str = "Statistical Test Results",
) -> Table:
    """
    Generate a table of statistical test results.

    Args:
        tests: List of test result dicts
        caption: Table caption

    Returns:
        Table object
    """
    headers = ["Test", "Statistic", "p-value", "Significant", "Effect Size"]
    rows = []

    for test in tests:
        name = test.get("name", "Unknown")
        stat = test.get("statistic")
        p = test.get("p_value")
        sig = test.get("significant", False)
        effect = test.get("effect_size")

        stat_str = f"{stat:.3f}" if stat is not None else "—"
        p_str = f"{p:.4f}" if p is not None else "—"
        sig_str = "Yes" if sig else "No"
        effect_str = f"{effect:.3f}" if effect is not None else "—"

        rows.append([name, stat_str, p_str, sig_str, effect_str])

    return Table(
        name="test_table",
        caption=caption,
        headers=headers,
        rows=rows,
    )


# =============================================================================
# High-Level Manuscript Generation
# =============================================================================

def generate_manuscript(
    study_path: Path,
    paper_config: Optional[dict] = None,
) -> Manuscript:
    """
    Generate a complete manuscript from study results.

    Args:
        study_path: Path to study directory
        paper_config: Optional paper configuration

    Returns:
        Manuscript object
    """
    # Load study config
    config = read_yaml(study_path / "config.yaml")
    study_config = config.get("study", {})

    # Load results
    analyze_path = study_path / "stages" / "5_analyze"
    aggregates = json_load(analyze_path / "aggregates.json")
    tests = json_load(analyze_path / "tests.json")

    # Load trial info
    generate_path = study_path / "stages" / "2_generate"
    trials = json_load(generate_path / "trials.json")
    n_trials = len(trials)
    n_tasks = len(set(t.get("task_id") for t in trials))

    # Extract metadata
    hypothesis = study_config.get("hypothesis", "")
    research_questions = config.get("research_questions", [])
    alpha = config.get("analysis", {}).get("alpha", 0.05)

    # Paper metadata
    if paper_config:
        title = paper_config.get("paper", {}).get("title", study_config.get("name", "Study"))
        authors = paper_config.get("paper", {}).get("authors", [])
    else:
        title = study_config.get("name", "Study")
        authors = []

    # Check if significant
    significant = any(t.get("significant", False) for t in tests)
    effect_size = tests[0].get("effect_size") if tests else None

    # Generate sections
    sections = [
        generate_introduction(hypothesis, research_questions),
        generate_methods(config, n_trials, n_tasks),
        generate_results(aggregates, tests, alpha),
        generate_discussion(hypothesis, significant, effect_size),
    ]

    # Generate tables
    tables = [
        generate_results_table(aggregates),
        generate_test_table(tests),
    ]

    # Generate abstract
    conditions = list(aggregates.keys())
    if len(conditions) >= 2:
        c1, c2 = conditions[0], conditions[1]
        rate1 = aggregates[c1].get("rate", 0)
        rate2 = aggregates[c2].get("rate", 0)
        abstract = (
            f"We investigated {hypothesis.lower().rstrip('.')}. "
            f"Results showed {c1} achieved {rate1:.1%} accuracy "
            f"compared to {rate2:.1%} for {c2}. "
            f"This difference was {'statistically significant' if significant else 'not statistically significant'}."
        )
    else:
        abstract = f"We investigated {hypothesis.lower().rstrip('.')}."

    return Manuscript(
        title=title,
        authors=authors,
        abstract=abstract,
        sections=sections,
        tables=tables,
        metadata={
            "generated": now_iso(),
            "study_path": str(study_path),
        },
    )


def save_manuscript(
    manuscript: Manuscript,
    output_dir: Path,
    formats: list[str] = ["markdown"],
) -> dict[str, Path]:
    """
    Save manuscript to files.

    Args:
        manuscript: Manuscript object
        output_dir: Output directory
        formats: List of formats to save (markdown, latex)

    Returns:
        Dict mapping format to output path
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {}

    if "markdown" in formats:
        md_path = output_dir / "manuscript.md"
        md_path.write_text(manuscript.to_markdown())
        outputs["markdown"] = md_path

    # LaTeX would require more complex templating
    # For now, just save markdown

    return outputs
