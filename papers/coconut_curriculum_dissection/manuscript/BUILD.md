# Manuscript Build Instructions

## Prerequisites

- [Pandoc](https://pandoc.org/) 3.x+
- XeLaTeX (via TeX Live or similar)
- Python 3.10+
- Times New Roman font installed

## Output Formats

| Format | Command |
|--------|---------|
| PDF (ICML two-column) | See below |
| HTML | `pandoc manuscript.md -o output/manuscript.html -s` |
| DOCX | `pandoc manuscript.md -o output/manuscript.docx` |

## ICML-Format PDF Build

The PDF uses an ICML-style two-column layout. Because pandoc's default LaTeX output uses `longtable` (incompatible with two-column mode), a post-processing step patches the generated LaTeX before compilation.

**All commands run from this directory** (`papers/coconut_curriculum_dissection/manuscript/`).

### Step 1: Generate LaTeX from Markdown

```bash
pandoc manuscript.md -o output/manuscript.tex \
  -V documentclass=article \
  -V fontsize=10pt \
  -V mainfont="Times New Roman" \
  -V geometry:top=1in \
  -V geometry:bottom=1in \
  -V geometry:left=1in \
  -V geometry:right=1in \
  -V geometry:columnsep=0.25in \
  -V pagestyle=plain \
  -H conference_preamble.tex \
  -s
```

### Step 2: Post-process for two-column compatibility

```bash
python3 fix_tables.py
```

This script (`fix_tables.py`) performs the following transformations:

- Replaces `longtable` environments with `tabular` + `resizebox` (twocolumn compatibility)
- Wraps title, author, and abstract in a full-width `\twocolumn[...]` block (ICML style)
- Promotes pandoc's shifted heading levels (`\subsection` -> `\section`, etc.)
- Makes wide tables (8+ columns) span both columns via `table*`
- Makes key figures (OOD bar chart, training curves) span both columns via `figure*`
- Removes the table of contents

### Step 3: Compile PDF (run twice for cross-references)

```bash
xelatex -interaction=nonstopmode -output-directory=output output/manuscript.tex
xelatex -interaction=nonstopmode -output-directory=output output/manuscript.tex
```

### One-liner

```bash
pandoc manuscript.md -o output/manuscript.tex -V documentclass=article -V fontsize=10pt -V mainfont="Times New Roman" -V geometry:top=1in -V geometry:bottom=1in -V geometry:left=1in -V geometry:right=1in -V geometry:columnsep=0.25in -V pagestyle=plain -H conference_preamble.tex -s && python3 fix_tables.py && xelatex -interaction=nonstopmode -output-directory=output output/manuscript.tex && xelatex -interaction=nonstopmode -output-directory=output output/manuscript.tex
```

### Clean up intermediate files

```bash
rm -f output/manuscript.tex output/manuscript.aux output/manuscript.log output/manuscript.out output/manuscript.toc
```

## ICML Format Specifications

| Parameter | Value |
|-----------|-------|
| Font | 10pt Times New Roman |
| Text width | 6.5in |
| Text height | 9.0in |
| Margins | 1in all sides |
| Column separation | 0.25in |
| Title/author/abstract | Full-width (single column) |
| Body | Two-column, starting from Section 1 |

## Files

| File | Purpose |
|------|---------|
| `manuscript.md` | Source manuscript (Markdown) |
| `conference_preamble.tex` | LaTeX preamble for conference styling (spacing, captions, lists) |
| `fix_tables.py` | Post-processor: longtable->tabular, ICML title block, heading levels |
| `figures/` | Source figures (PNG) |
| `output/` | Generated outputs (PDF, HTML, DOCX) |
