# False Belief (Belief Bias) Dataset Analysis

This document summarizes the structure, experimental design, and analysis of the belief bias dataset.

## Overview

Belief bias is the tendency to judge the logical validity of an argument based on the believability of its conclusion rather than the actual logical structure. People tend to:
- **Accept** invalid arguments if the conclusion "sounds right" (believable)
- **Reject** valid arguments if the conclusion "sounds wrong" (unbelievable)

## Experimental Design (2×2×2)

### Dimensions

1. **Logical Validity**: Valid vs Invalid syllogisms
2. **Conclusion Believability**: Believable vs Unbelievable conclusions
3. **Object Type**: Real-life Objects (Treatment) vs Nonsense Words (Control)

### Dataset Files

- **Treatment**: `Data/false_belief/all_permutations/t_[1, 2, 3, 4, 5, 6, 7]_dm_full_Treatment.json`
- **Control**: `Data/false_belief/all_permutations/t_[1, 2, 3, 4, 5, 6, 7]_dm_full_Control.json`

### Example Syllogisms

**Valid + Believable (Real-life):**
```
Premise 1: Some librarians are happy people.
Premise 2: All happy people are healthy people.
Conclusion: Some librarians are healthy people.
→ Correct answer: Valid/Yes (logic is sound, conclusion believable)
```

**Invalid + Believable (Real-life):**
```
Premise 1: All librarians are happy people.
Premise 2: Some happy people are healthy people.
Conclusion: Some librarians are healthy people.
→ Correct answer: Invalid/No (logic is flawed, but conclusion "sounds right")
```

**Invalid + Unbelievable (Real-life):**
```
Premise 1: All happy people are healthy people.
Premise 2: All happy people are librarians.
Conclusion: All healthy people are librarians.
→ Correct answer: Invalid/No (logic is flawed, conclusion "sounds wrong")
```

**Control (Nonsense words):**
```
Premise 1: All nept are bript.
Premise 2: All nept are jeft.
Conclusion: All bript are jeft.
→ Only logic matters, no real-world believability to influence judgment
```

## Results for OLMo-3-7B-Instruct

### Acceptance Rates

"Acceptance rate" = percentage of arguments the model accepts as logically valid.

| Condition | Real-life Objects | Nonsense Words | Correct Rate |
|-----------|------------------|----------------|--------------|
| Valid-Believable | 99.4% | 94.6% | 100% |
| Valid-Unbelievable | 81.2% | 100.0% | 100% |
| **Invalid-Believable** | **93.8%** | 82.0% | **0%** |
| Invalid-Unbelievable | 17.8% | 55.6% | 0% |

### Key Finding: Massive Belief Bias

For **Invalid + Believable** arguments with real-life objects:
- Model accepts **93.8%** as valid (should be 0%)
- The believable conclusion overrides logical analysis

For **Invalid + Unbelievable** arguments:
- Model correctly rejects most (82.2% rejection rate)
- The unbelievable conclusion helps the model identify invalidity

## Two Ways to Measure Belief Bias

### 1. Paper's Methodology

The paper compares "consistent" conditions (where belief and logic align) to "neutral" baselines:

```
Belief Valid = P(accept | valid, believable, real-life) − P(accept | valid, non-real)
             = 99.4% − 97.3% = +2.1%

Belief Invalid = P(accept | invalid, unbelievable, real-life) − P(accept | invalid, non-real)
               = 17.8% − 68.6% = −50.8%
```

**Interpretation:**
- **Belief Valid (+2.1%)**: Minimal effect; model handles valid arguments similarly regardless of believability
- **Belief Invalid (−50.8%)**: Large negative effect; unbelievable conclusions help the model reject invalid arguments

### 2. Classic Psychology Definition

Compares believable vs unbelievable within invalid arguments:

```
Classic Bias = P(accept | invalid, believable) − P(accept | invalid, unbelievable)

Real-life Objects: 93.8% − 17.8% = 76.0%
Nonsense Words:    82.0% − 55.6% = 26.4%
```

**Interpretation:**
- OLMo shows **76% belief bias** with real-life objects
- This exceeds typical human belief bias (~50% in classic studies)

### Reconciling Both Metrics

| Metric | Value | What It Measures |
|--------|-------|------------------|
| Paper's Belief Invalid | −50.8% | Unbelievable conclusions aid rejection (helpful) |
| Classic Invalid Bias | +76.0% | Believable conclusions cause false acceptance (harmful) |

Both metrics reveal the same underlying phenomenon: **believability strongly influences the model's judgments**. The paper's metric captures how unbelievability helps; the classic metric captures how believability hurts.

## The Problematic Case

The paper's methodology focuses on "consistent" conditions where belief and logic align. However, the most problematic case is the **inconsistent** condition:

**Invalid + Believable (Real-life Objects):**
- Acceptance rate: 93.8%
- This should be 0%
- The model accepts nearly all logically invalid arguments when the conclusion sounds plausible

This represents a fundamental failure to separate logical validity from real-world plausibility.

## Control Condition Analysis

Nonsense words should eliminate belief bias since there's no real-world meaning. However:

- Invalid arguments still accepted 56-82% of the time
- Model struggles with syllogistic reasoning even without belief influence
- "Believability" for nonsense words may reflect structural patterns

This suggests OLMo has **two problems**:
1. Belief bias when real-world content is present
2. Poor syllogistic reasoning ability in general

## Comparison with Human Data

From Markovits & Nantel (1989), humans show:
- Invalid-Believable: ~70% acceptance
- Invalid-Unbelievable: ~20% acceptance
- Human belief bias: ~50%

**OLMo-3-7B-Instruct shows stronger belief bias (76%) than typical humans.**

## Analysis Output Files

| File | Contents |
|------|----------|
| `_[...]_.csv` | Summary: acceptance rates by condition |
| `_[...]_confidences.csv` | 95% bootstrapped confidence intervals |
| `_[...]_full_answers.csv` | Per-sample predictions with validity/believability labels |
| `logging_aux_[...].txt` | Detailed breakdown by template |

## Code References

- Data generation: `Data_generation/generate_samples_false_belief.py`
- Templates: `Data_generation/templates.py` (see `ALL_FALSE_BELIEF_DEEPMIND_TEMP`)
- Analysis: `Analysis/fb_analysis.py`
- Answer parsing: `Analysis/analyze.py`
