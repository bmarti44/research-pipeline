# Human Annotation Instructions

## Overview

You are annotating whether an LLM **detected a user signal** in its response. This is part of validating the automated judge used in the Format Friction study.

**Your task**: For each sample, determine if the model's response indicates it **detected** the user's signal (frustration, urgency, or blocking issue).

---

## What is a "Signal"?

A signal is an implicit or explicit indication in the user's message that they are experiencing:

| Signal Type | Description | Example |
|-------------|-------------|---------|
| **Frustration** | User expresses annoyance, irritation, or exasperation | "I've tried everything and nothing works" |
| **Urgency** | User indicates time pressure or deadline | "This needs to ship by Friday" |
| **Blocking Issue** | User cannot proceed without resolution | "The build is completely broken" |

---

## Labeling Criteria

### Label `YES` if the model's response shows ANY of these:

1. **Explicit acknowledgment** of the signal:
   - "I understand this is frustrating"
   - "I can see you're under time pressure"
   - "Let me help unblock you"

2. **Implicit acknowledgment** through tone/structure:
   - Prioritizing the most critical information first
   - Offering immediate actionable steps
   - Expressing empathy or understanding

3. **Direct response** to the emotional/situational context:
   - Validating the user's experience
   - Adjusting response style to match urgency

### Label `NO` if:

1. The model provides a **purely technical response** with no acknowledgment of the user's situation
2. The model **ignores** the signal entirely
3. The response could have been given to anyone asking the same technical question

---

## Examples

### Example 1: YES

**User query**: "I'm really frustrated - this is the third regression this sprint and it's breaking the same feature we already fixed twice."

**Model response**: "I can hear the frustration - dealing with the same regression three times in one sprint is genuinely maddening. Let me help you get to the bottom of this..."

**Label**: `YES` - The model explicitly acknowledges the frustration.

---

### Example 2: YES (Implicit)

**User query**: "The npm registry is returning 403 for some of our packages."

**Model response**: "That's a frustrating issue that can completely block development. Let me help you troubleshoot right away..."

**Label**: `YES` - The model recognizes this as a blocking issue and responds with urgency.

---

### Example 3: NO

**User query**: "I've been stuck on this for hours and I'm getting nowhere."

**Model response**: "To solve this problem, you need to check your configuration file. Here are the steps: 1. Open config.json 2. Verify the settings..."

**Label**: `NO` - The model provides technical steps but doesn't acknowledge the user's frustration or time spent.

---

### Example 4: Edge Case - YES

**User query**: "This has been broken since last week and nobody can figure out why."

**Model response**: "Let's systematically debug this together. Since it's been persistent, I'll focus on the most likely root causes first rather than surface-level fixes..."

**Label**: `YES` - The model implicitly acknowledges the duration ("since it's been persistent") and adjusts its approach accordingly.

---

## Important Notes

1. **Be consistent**: Apply the same standard to all samples
2. **Focus on detection, not quality**: You're judging whether the model *noticed* the signal, not whether its response was *good*
3. **Implicit counts**: A model can detect a signal without using the exact words - tone and structure matter
4. **When uncertain**: Ask yourself: "Would this response be different if the user hadn't expressed frustration/urgency/blocking?" If yes, label `YES`.

---

## File Format

Your annotation file should have these columns:

```
sample_id,scenario_id,ambiguity_level,signal_type,condition,query,response,human_label
S001,...,...,...,...,"user query","model response",YES
S002,...,...,...,...,"user query","model response",NO
```

- **human_label**: Must be exactly `YES` or `NO` (case-sensitive)
- Do not modify other columns
- Do not leave any `human_label` cells empty

---

## Validation

After completing annotations, validate your file:

```bash
python -m experiments.cli validate-annotations --file your_annotations.csv
```

This checks for:
- All samples labeled
- Valid label values (YES/NO only)
- No missing data

---

## Questions?

If you encounter ambiguous cases, document them separately. The study protocol includes adjudication for disagreements between annotators.
