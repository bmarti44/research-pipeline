"""Post-process pandoc LaTeX for ICML-style two-column format.

Strategy: pandoc generates a ONECOLUMN document (no twocolumn class option).
This script:
1. Styles title/author centered and full-width
2. Formats abstract with bold label and indented block
3. Inserts \\twocolumn before §1 Introduction
4. Replaces longtable with tabular+resizebox
5. Makes wide tables and key figures span both columns
"""
import re

with open("output/manuscript.tex", "r") as f:
    tex = f.read()

# --- Preamble fixes ---

tex = tex.replace(r"\usepackage{longtable,booktabs,array}",
                   r"\usepackage{booktabs,array}")

tex = re.sub(
    r"\\makeatletter\n\\patchcmd\\longtable.*?\\makeatother\n",
    "", tex, flags=re.DOTALL
)
tex = re.sub(r"\\makesavenoteenv\{longtable\}\n", "", tex)

# Remove TOC
tex = re.sub(
    r"\{\s*\n\\setcounter\{tocdepth\}\{3\}\s*\n\\tableofcontents\s*\n\}",
    "", tex
)

# --- Title: centered, large, bold ---
tex = re.sub(
    r"\\section\{(The Curriculum Is the Mechanism:.*?)\}"
    r"\\label\{the-curriculum-is-the-mechanism.*?\}",
    r"\\begin{center}\n"
    r"{\\LARGE\\bfseries \\1\\par}\n"
    r"\\vspace{0.8em}\n"
    r"\\end{center}",
    tex
)

# --- Author: centered ---
tex = tex.replace(
    r"\textbf{Anonymous Author(s)} Anonymous Institution",
    "\\begin{center}\n"
    "{\\large Anonymous Author(s)}\\par\n"
    "\\vspace{0.3em}\n"
    "{\\normalsize Anonymous Institution}\n"
    "\\end{center}\n"
    "\\vspace{0.8em}"
)

# --- Abstract: bold label + quote-style indented block ---
# The abstract is the single paragraph starting with "COCONUT's training
# curriculum has not been isolated" through "drives COCONUT's accuracy on ProsQA."
# It sits between the author block and "## 1 Introduction" in the source.

# In the generated LaTeX, the abstract paragraph starts after \vspace{0.5em}
# and ends before \subsection{1 Introduction} (or \section{1 Introduction}).
# Find it and wrap with formatting.

abstract_start = tex.find("COCONUT\\textquotesingle s training curriculum")
if abstract_start == -1:
    abstract_start = tex.find("COCONUT's training curriculum")
if abstract_start == -1:
    abstract_start = tex.find("COCONUT")

# Find end of abstract paragraph (ends with "on ProsQA." followed by blank line)
abstract_end_match = re.search(
    r"on\s+ProsQA\.\s*\n",
    tex[abstract_start:]
)

if abstract_start >= 0 and abstract_end_match:
    abs_end = abstract_start + abstract_end_match.end()
    abstract_text = tex[abstract_start:abs_end].strip()

    # Wrap in a formatted block
    formatted_abstract = (
        "\\vspace{0.5em}\n"
        "\\noindent\\textbf{Abstract.} "
        + abstract_text + "\n"
        "\\vspace{1em}\n"
    )
    tex = tex[:abstract_start] + formatted_abstract + tex[abs_end:]

# --- Use \twocolumn[...title block...] for ICML-style layout ---
# This places the title/abstract full-width at the top of page 1,
# then starts two-column immediately below on the same page.
#
# We need to:
# 1. Remove the title block from its current position
# 2. Wrap it in \twocolumn[...] right after \begin{document}

# Find everything between \begin{document} and \subsection{1 Introduction}
doc_start_match = re.search(r"\\begin\{document\}\s*\n", tex)
intro_match = re.search(r"\\subsection\{1 Introduction\}", tex)

if doc_start_match and intro_match:
    doc_body_start = doc_start_match.end()
    intro_pos = intro_match.start()

    # Extract the title block
    title_block = tex[doc_body_start:intro_pos].strip()

    # Add vspace at the end of the title block for separation
    title_block += "\n\\vspace{1em}"

    # Replace: remove title block from its position, insert \twocolumn[...] after \begin{document}
    tex = (tex[:doc_body_start]
           + "\n\\twocolumn[\n"
           + title_block + "\n"
           + "]\n\n"
           + "\\section{1 Introduction}\\label{introduction}\n"
           + tex[intro_match.end():]
           )

# Also upgrade all other \subsection{N ...} to \section{N ...} and
# \subsubsection{N.M ...} to \subsection{N.M ...} since the heading
# levels are shifted by pandoc treating the title as \section.

# Main sections: \subsection{N ...} -> \section{N ...}
tex = re.sub(
    r"\\subsection\{(\d+ [^}]+)\}",
    lambda m: "\\section{" + m.group(1) + "}",
    tex
)

# Subsections: \subsubsection{N.M ...} -> \subsection{N.M ...}
tex = re.sub(
    r"\\subsubsection\{(\d+\.\d+ [^}]+)\}",
    lambda m: "\\subsection{" + m.group(1) + "}",
    tex
)

# Appendix sections: \subsubsection{A.N ...} -> \subsection{A.N ...}
tex = re.sub(
    r"\\subsubsection\{(A\.\d+ [^}]+)\}",
    lambda m: "\\subsection{" + m.group(1) + "}",
    tex
)

# "Appendix" heading
tex = tex.replace("\\subsection{Appendix}", "\\section{Appendix}")

# "References" heading
tex = tex.replace("\\subsection{References}", "\\section{References}")

# --- Table processing ---

table_counter = 0
lines = tex.split("\n")
output_lines = []
in_longtable = False
longtable_lines = []

for line in lines:
    if r"\begin{longtable}" in line:
        in_longtable = True
        longtable_lines = [line]
    elif r"\end{longtable}" in line and in_longtable:
        longtable_lines.append(line)
        table_counter += 1
        block = "\n".join(longtable_lines)

        col_match = re.search(r"\{([>@<lcr{}\\.\s\{\}]+)\}", longtable_lines[0])
        n_cols = 0
        if col_match:
            spec = col_match.group(1)
            n_cols = sum(1 for c in spec if c in "lcr")

        block = block.replace(r"\begin{longtable}[]{", r"\begin{tabular}{")
        block = block.replace(r"\end{longtable}", r"\end{tabular}")

        cleaned = []
        for bline in block.split("\n"):
            stripped = bline.strip()
            if stripped in (r"\endhead", r"\endfirsthead",
                           r"\endfoot", r"\endlastfoot"):
                continue
            cleaned.append(bline)
        block = "\n".join(cleaned)

        is_wide = n_cols >= 8

        if is_wide:
            wrapped = (
                r"\begin{table*}[htbp]" + "\n"
                + r"\centering" + "\n"
                + r"\small" + "\n"
                + block + "\n"
                + r"\end{table*}"
            )
        else:
            wrapped = (
                r"\begin{table}[htbp]" + "\n"
                + r"\centering" + "\n"
                + r"\resizebox{\columnwidth}{!}{%" + "\n"
                + block + "\n"
                + "}" + "\n"
                + r"\end{table}"
            )
        output_lines.append(wrapped)
        in_longtable = False
        longtable_lines = []
    elif in_longtable:
        longtable_lines.append(line)
    else:
        output_lines.append(line)

tex = "\n".join(output_lines)

# --- Figures: make key figures span both columns ---

tex = re.sub(
    r"\\begin\{figure\}(\[htbp\])?\s*\n(.*?Out-of-distribution accuracy.*?)\\end\{figure\}",
    lambda m: m.group(0).replace(r"\begin{figure}", r"\begin{figure*}").replace(
        r"\end{figure}", r"\end{figure*}"
    ),
    tex, count=1, flags=re.DOTALL
)

tex = re.sub(
    r"\\begin\{figure\}(\[htbp\])?\s*\n(.*?Training curves.*?)\\end\{figure\}",
    lambda m: m.group(0).replace(r"\begin{figure}", r"\begin{figure*}").replace(
        r"\end{figure}", r"\end{figure*}"
    ),
    tex, count=1, flags=re.DOTALL
)

tex = re.sub(
    r"\\begin\{figure\}(\[htbp\])?\s*\n(.*?Progressive forward corruption accuracy.*?)\\end\{figure\}",
    lambda m: m.group(0).replace(r"\begin{figure}", r"\begin{figure*}").replace(
        r"\end{figure}", r"\end{figure*}"
    ),
    tex, count=1, flags=re.DOTALL
)

tex = re.sub(
    r"\\begin\{figure\}(\[htbp\])?\s*\n(.*?Linear probe accuracy by layer and thought position.*?)\\end\{figure\}",
    lambda m: m.group(0).replace(r"\begin{figure}", r"\begin{figure*}").replace(
        r"\end{figure}", r"\end{figure*}"
    ),
    tex, count=1, flags=re.DOTALL
)

with open("output/manuscript.tex", "w") as f:
    f.write(tex)

print(f"Done. {table_counter} tables. Full-width title/abstract, twocolumn from §1.")
