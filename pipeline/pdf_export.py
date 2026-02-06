"""
PDF export functionality for the research pipeline.

Provides functions for:
- Converting markdown to PDF
- Exporting manuscripts as PDF
- Creating PDF reports from analysis results
- Generating PDF figures and tables
"""

import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Any
import shutil

from .utils import ensure_dir, now_iso


# =============================================================================
# PDF Export Configuration
# =============================================================================

@dataclass
class PDFConfig:
    """Configuration for PDF export."""
    page_size: str = "letter"  # letter, a4
    margin_top: str = "1in"
    margin_bottom: str = "1in"
    margin_left: str = "1in"
    margin_right: str = "1in"
    font_size: int = 11
    font_family: str = "serif"
    line_spacing: float = 1.5
    include_toc: bool = True
    include_page_numbers: bool = True
    header: Optional[str] = None
    footer: Optional[str] = None


# =============================================================================
# Markdown to PDF Conversion
# =============================================================================

def check_pandoc_available() -> bool:
    """Check if pandoc is available on the system."""
    return shutil.which("pandoc") is not None


def check_latex_available() -> bool:
    """Check if LaTeX (pdflatex) is available on the system."""
    return shutil.which("pdflatex") is not None


def markdown_to_pdf(
    markdown_path: Path,
    output_path: Path,
    config: Optional[PDFConfig] = None,
    title: Optional[str] = None,
    author: Optional[str] = None,
) -> bool:
    """
    Convert a markdown file to PDF using pandoc.

    Args:
        markdown_path: Path to markdown file
        output_path: Path for output PDF
        config: PDF configuration options
        title: Document title
        author: Document author

    Returns:
        True if successful, False otherwise
    """
    if not check_pandoc_available():
        raise RuntimeError("pandoc is required for PDF export. Install from: https://pandoc.org/")

    config = config or PDFConfig()

    # Build pandoc command
    cmd = [
        "pandoc",
        str(markdown_path),
        "-o", str(output_path),
        "--pdf-engine=pdflatex" if check_latex_available() else "--pdf-engine=wkhtmltopdf",
    ]

    # Add metadata
    if title:
        cmd.extend(["--metadata", f"title={title}"])
    if author:
        cmd.extend(["--metadata", f"author={author}"])

    # Add geometry options
    geometry = f"margin={config.margin_top}"
    cmd.extend(["-V", f"geometry:{geometry}"])

    # Add font size
    cmd.extend(["-V", f"fontsize={config.font_size}pt"])

    # Add table of contents
    if config.include_toc:
        cmd.append("--toc")

    # Add page numbers
    if config.include_page_numbers:
        cmd.extend(["-V", "pagestyle=plain"])

    # Run pandoc
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
        )
        return True
    except subprocess.CalledProcessError as e:
        print(f"PDF export failed: {e.stderr}")
        return False
    except FileNotFoundError:
        print("pandoc not found. Please install pandoc for PDF export.")
        return False


def markdown_string_to_pdf(
    markdown_content: str,
    output_path: Path,
    config: Optional[PDFConfig] = None,
    title: Optional[str] = None,
    author: Optional[str] = None,
) -> bool:
    """
    Convert markdown string to PDF.

    Args:
        markdown_content: Markdown content as string
        output_path: Path for output PDF
        config: PDF configuration options
        title: Document title
        author: Document author

    Returns:
        True if successful, False otherwise
    """
    # Write to temporary file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
        f.write(markdown_content)
        temp_path = Path(f.name)

    try:
        return markdown_to_pdf(temp_path, output_path, config, title, author)
    finally:
        temp_path.unlink()


# =============================================================================
# HTML to PDF Conversion (Fallback)
# =============================================================================

def check_wkhtmltopdf_available() -> bool:
    """Check if wkhtmltopdf is available."""
    return shutil.which("wkhtmltopdf") is not None


def html_to_pdf(
    html_path: Path,
    output_path: Path,
    config: Optional[PDFConfig] = None,
) -> bool:
    """
    Convert HTML file to PDF using wkhtmltopdf.

    Args:
        html_path: Path to HTML file
        output_path: Path for output PDF
        config: PDF configuration options

    Returns:
        True if successful, False otherwise
    """
    if not check_wkhtmltopdf_available():
        raise RuntimeError("wkhtmltopdf is required. Install from: https://wkhtmltopdf.org/")

    config = config or PDFConfig()

    cmd = [
        "wkhtmltopdf",
        "--page-size", config.page_size.upper(),
        "--margin-top", config.margin_top,
        "--margin-bottom", config.margin_bottom,
        "--margin-left", config.margin_left,
        "--margin-right", config.margin_right,
    ]

    if config.include_page_numbers:
        cmd.extend(["--footer-center", "[page] / [topage]"])

    cmd.extend([str(html_path), str(output_path)])

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
        )
        return True
    except subprocess.CalledProcessError as e:
        print(f"PDF export failed: {e.stderr}")
        return False
    except FileNotFoundError:
        print("wkhtmltopdf not found.")
        return False


# =============================================================================
# Simple HTML Generation (for fallback PDF)
# =============================================================================

def markdown_to_html(markdown_content: str, title: str = "Document") -> str:
    """
    Convert markdown to simple HTML.

    Args:
        markdown_content: Markdown content
        title: HTML title

    Returns:
        HTML string
    """
    # Try to use markdown library if available
    try:
        import markdown
        body = markdown.markdown(
            markdown_content,
            extensions=["tables", "fenced_code"],
        )
    except ImportError:
        # Fallback: basic conversion
        body = basic_markdown_to_html(markdown_content)

    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>{title}</title>
    <style>
        body {{
            font-family: Georgia, serif;
            font-size: 11pt;
            line-height: 1.6;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
        }}
        h1, h2, h3 {{ font-family: Helvetica, Arial, sans-serif; }}
        table {{
            border-collapse: collapse;
            width: 100%;
            margin: 1em 0;
        }}
        th, td {{
            border: 1px solid #ddd;
            padding: 8px;
            text-align: left;
        }}
        th {{ background-color: #f5f5f5; }}
        code {{
            background-color: #f5f5f5;
            padding: 2px 4px;
            border-radius: 3px;
        }}
        pre {{
            background-color: #f5f5f5;
            padding: 10px;
            border-radius: 5px;
            overflow-x: auto;
        }}
    </style>
</head>
<body>
{body}
</body>
</html>"""
    return html


def basic_markdown_to_html(markdown: str) -> str:
    """
    Basic markdown to HTML conversion without dependencies.

    Args:
        markdown: Markdown content

    Returns:
        HTML string (basic conversion)
    """
    import re

    html = markdown

    # Headers
    html = re.sub(r"^### (.+)$", r"<h3>\1</h3>", html, flags=re.MULTILINE)
    html = re.sub(r"^## (.+)$", r"<h2>\1</h2>", html, flags=re.MULTILINE)
    html = re.sub(r"^# (.+)$", r"<h1>\1</h1>", html, flags=re.MULTILINE)

    # Bold and italic
    html = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", html)
    html = re.sub(r"\*(.+?)\*", r"<em>\1</em>", html)

    # Code blocks
    html = re.sub(r"```(\w*)\n(.*?)```", r"<pre><code>\2</code></pre>", html, flags=re.DOTALL)

    # Inline code
    html = re.sub(r"`(.+?)`", r"<code>\1</code>", html)

    # Lists
    html = re.sub(r"^- (.+)$", r"<li>\1</li>", html, flags=re.MULTILINE)

    # Paragraphs
    paragraphs = html.split("\n\n")
    processed = []
    for p in paragraphs:
        p = p.strip()
        if p and not p.startswith("<"):
            p = f"<p>{p}</p>"
        processed.append(p)
    html = "\n\n".join(processed)

    return html


# =============================================================================
# Report PDF Export
# =============================================================================

def export_results_pdf(
    study_path: Path,
    output_path: Optional[Path] = None,
    config: Optional[PDFConfig] = None,
) -> Path:
    """
    Export study results as PDF.

    Args:
        study_path: Path to study directory
        output_path: Output PDF path (default: outputs/RESULTS.pdf)
        config: PDF configuration

    Returns:
        Path to generated PDF
    """
    # Default output path
    if output_path is None:
        output_path = study_path / "outputs" / "RESULTS.pdf"

    ensure_dir(output_path.parent)

    # Look for RESULTS.md
    results_md = study_path / "outputs" / "RESULTS.md"
    if not results_md.exists():
        raise FileNotFoundError(f"RESULTS.md not found at {results_md}")

    # Try pandoc first
    if check_pandoc_available():
        success = markdown_to_pdf(results_md, output_path, config)
        if success:
            return output_path

    # Fallback to wkhtmltopdf via HTML
    if check_wkhtmltopdf_available():
        markdown_content = results_md.read_text()
        html_content = markdown_to_html(markdown_content, "Study Results")

        with tempfile.NamedTemporaryFile(mode="w", suffix=".html", delete=False) as f:
            f.write(html_content)
            temp_html = Path(f.name)

        try:
            success = html_to_pdf(temp_html, output_path, config)
            if success:
                return output_path
        finally:
            temp_html.unlink()

    raise RuntimeError(
        "No PDF converter available. Install pandoc (recommended) or wkhtmltopdf."
    )


def export_manuscript_pdf(
    manuscript_path: Path,
    output_path: Path,
    config: Optional[PDFConfig] = None,
    title: Optional[str] = None,
    author: Optional[str] = None,
) -> Path:
    """
    Export manuscript markdown as PDF.

    Args:
        manuscript_path: Path to manuscript markdown file
        output_path: Output PDF path
        config: PDF configuration
        title: Document title
        author: Document author

    Returns:
        Path to generated PDF
    """
    if not manuscript_path.exists():
        raise FileNotFoundError(f"Manuscript not found: {manuscript_path}")

    ensure_dir(output_path.parent)

    success = markdown_to_pdf(manuscript_path, output_path, config, title, author)
    if success:
        return output_path

    raise RuntimeError("PDF export failed")


# =============================================================================
# Batch Export
# =============================================================================

def export_all_studies_pdf(
    paper_path: Path,
    output_dir: Optional[Path] = None,
) -> dict[str, Path]:
    """
    Export all studies in a paper to PDF.

    Args:
        paper_path: Path to paper directory
        output_dir: Output directory (default: paper_path/outputs)

    Returns:
        Dict mapping study name to PDF path
    """
    if output_dir is None:
        output_dir = paper_path / "outputs"

    ensure_dir(output_dir)

    studies_dir = paper_path / "studies"
    if not studies_dir.exists():
        return {}

    results = {}
    for study_dir in studies_dir.iterdir():
        if not study_dir.is_dir():
            continue

        study_name = study_dir.name
        results_md = study_dir / "outputs" / "RESULTS.md"

        if results_md.exists():
            try:
                pdf_path = export_results_pdf(
                    study_dir,
                    output_dir / f"{study_name}_results.pdf",
                )
                results[study_name] = pdf_path
                print(f"Exported {study_name}: {pdf_path}")
            except Exception as e:
                print(f"Failed to export {study_name}: {e}")

    return results


# =============================================================================
# Utility Functions
# =============================================================================

def get_available_pdf_engines() -> list[str]:
    """
    Get list of available PDF conversion tools.

    Returns:
        List of available engine names
    """
    engines = []
    if check_pandoc_available():
        engines.append("pandoc")
    if check_latex_available():
        engines.append("pdflatex")
    if check_wkhtmltopdf_available():
        engines.append("wkhtmltopdf")
    return engines


def print_pdf_setup_instructions() -> None:
    """Print instructions for setting up PDF export."""
    print("PDF Export Setup")
    print("=" * 50)
    print()

    engines = get_available_pdf_engines()
    if engines:
        print(f"Available engines: {', '.join(engines)}")
    else:
        print("No PDF engines found!")

    print()
    print("Installation options:")
    print()
    print("1. Pandoc (recommended):")
    print("   macOS: brew install pandoc")
    print("   Ubuntu: sudo apt install pandoc")
    print("   Windows: https://pandoc.org/installing.html")
    print()
    print("2. LaTeX (for best quality with pandoc):")
    print("   macOS: brew install --cask mactex")
    print("   Ubuntu: sudo apt install texlive-latex-base")
    print()
    print("3. wkhtmltopdf (alternative):")
    print("   macOS: brew install wkhtmltopdf")
    print("   Ubuntu: sudo apt install wkhtmltopdf")
    print("   Windows: https://wkhtmltopdf.org/downloads.html")
