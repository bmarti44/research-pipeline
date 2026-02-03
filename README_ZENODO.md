# Format Friction: The Compliance Gap in Prompt-Based Tool Calling

## Overview

This archive contains the paper, code, and data for "Format Friction: The Compliance Gap in Prompt-Based Tool Calling" by Brian Martin and Stephen Lipmann.

## Key Finding

In prompt-based tool-calling systems, we observe a **10.3 percentage point gap** between signal detection (85.6%) and format compliance (75.3%). This "format friction" represents cases where the model recognized a signal but failed to produce the required XML structure.

**Key results:**
- Detection rates: 89.4% (NL) vs 85.6% (structured)
- Format friction: 10.3pp [95% CI: 5.8pp, 14.8pp]
- Ambiguity interaction: 0pp explicit vs 18.4pp implicit
- Two-pass recovery: 74% (Sonnet) and 50% (7B) of compliance gaps

## Contents

```
paper/
  FORMAT_FRICTION.md       - Main paper
  LIMITATIONS.md           - Detailed limitations discussion
  REVIEW.md               - Academic review and remediation
  figures/                - Publication figures (PNG format)

experiments/
  signal_detection_experiment.py  - Main experiment runner
  judge_scoring.py               - LLM judge implementation
  analyze_judged_results.py      - Statistical analysis
  two_pass_extraction.py         - Recovery mechanism
  remediation_analysis.py        - Additional analyses
  scenarios/signal_detection.py  - Scenario definitions (75 total)
  results/
    primary/              - Primary data files
    DATA_MANIFEST.md      - Data provenance documentation

requirements.txt          - Python dependencies
LICENSE                  - Licensing information (CC-BY-4.0, MIT, CC0)
CITATION.cff             - Citation metadata
.zenodo.json             - Zenodo metadata
```

## Reproduction

```bash
# Install dependencies
pip install -r requirements.txt

# Run analysis on primary data
python experiments/analyze_judged_results.py \
    experiments/results/primary/signal_detection_20260203_074411_judged.json

# Run remediation analysis (signal type, scenario variance, etc.)
python experiments/remediation_analysis.py \
    experiments/results/primary/signal_detection_20260203_074411_judged.json
```

## Primary Data Files

| File | Purpose | N |
|------|---------|---|
| `signal_detection_20260203_074411_judged.json` | Main results with judge scores | 750 trials |
| `signal_detection_20260203_121413.json` | Raw experiment data | 1500 observations |
| `two_pass_sonnet_nl_20260203_125603.json` | Sonnet recovery testing | 750 trials |
| `two_pass_qwen7b_nl_20260203_131141.json` | Qwen-7B recovery testing | 750 trials |

## Citation

```bibtex
@article{martin2026formatfriction,
  title={Format Friction: The Compliance Gap in Prompt-Based Tool Calling},
  author={Martin, Brian and Lipmann, Stephen},
  year={2026},
  note={Preprint}
}
```

## License

- Paper and figures: CC-BY-4.0
- Code: MIT License
- Data: CC0 (Public Domain)

## Contact

- Brian Martin: brian@brianmartin.com
- Stephen Lipmann: shlipmann@gmail.com
