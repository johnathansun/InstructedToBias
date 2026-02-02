# Certainty Effect Dataset Analysis

This document summarizes the structure and experimental design of the certainty effect bias dataset.

## Overview

The certainty effect tests whether models overweight certain outcomes compared to probabilistic ones. The dataset contains Treatment (with a certain option) and Control (both options risky) conditions.

## Dataset Structure

### Files

- **Treatment**: `Data/certainty/all_permutations/t_[1, 2, 3]_three_probs,two_probs_Treatment.json` (504 samples)
- **Control**: `Data/certainty/all_permutations/t_[1, 2, 3]_three_probs,two_probs_Control.json` (336 samples)

### Sample Structure

Each sample contains:

```json
{
    "template": 1,
    "subtemplates": {
        "bias_type_index": 1,
        "vals_index": 1,
        "options_text_template_id": 1,
        "options_a_template_id": 1,
        "options_b_template_id": 1,
        "permutation_index": 1
    },
    "option_a": { "option_text": "...", "option_type": "better_expected_value" },
    "option_b": { "option_text": "...", "option_type": "target" },
    "text": "Choose between:\nOption A - ...\nOption B - ...\nAnswer:",
    "human_or_right_answer": 1,
    "better_expected_value": 1,
    "target": 2
}
```

### Subtemplate Parameters

| Parameter | Values | Meaning |
|-----------|--------|---------|
| `bias_type_index` | 1, 2 | Experimental design type (see below) |
| `vals_index` | 1-7 | Which prize/probability value set |
| `options_text_template_id` | 1 | Option labels ("Option A", "Option B") |
| `options_a_template_id` | 1, 2 | Phrasing for risky option |
| `options_b_template_id` | 1, 2, 3 | Phrasing for target option |
| `permutation_index` | 1, 2 | Position (1=target second, 2=target first) |

### Distinct Combinations

2 × 7 × 1 × 2 × 2 × 2 = **112 unique subtemplate combinations**

Treatment has more samples (504 vs 336) because `options_b_template_id` has 3 variants for certain options ("with certainty", "for sure", "100% to win") vs 2 for risky options.

## Two Experimental Designs

### Design 1: Common Consequence Effect (`DEVIDE_OPTION_A_TO_THREE_PROBS`, bias_type_index=1)

Adds a common outcome to both options, making Option B certain.

**Control:**
```
Option A: $2500 @ 33%, $0 @ 67%
Option B: $2400 @ 34%, $0 @ 66%
```

**Treatment** (add $2400 @ 66% to both):
```
Option A: $2500 @ 33%, $2400 @ 66%, $0 @ 1%   ← 3 probability components
Option B: $2400 @ 100%                         ← becomes certain
```

**Properties:**
- Option A structure preserved (same top prize, same probability)
- Same transformation applied to both options
- Clean test of certainty effect
- Good for paired analysis

### Design 2: Common Ratio Effect (`DEVIDE_OPTION_A_TO_TWO_PROBS`, bias_type_index=2)

Scales all probabilities by a factor, making Option B certain.

**Control:**
```
Option A: $5000 @ 20%, $0 @ 80%   →  EV = $1,000
Option B: $3000 @ 25%, $0 @ 75%   →  EV = $750
```

**Treatment** (multiply probabilities by ~4):
```
Option A: $5000 @ 80%, $0 @ 20%   →  EV = $4,000
Option B: $3000 @ 100%            →  EV = $3,000
```

**Properties:**
- Option A probabilities completely different
- Expected values change dramatically
- Confounds certainty effect with probability magnitude
- Less suitable for paired analysis

## Pairing Treatment and Control

### Matching Key

Samples can be matched between Treatment and Control using:

```python
match_key = (
    bias_type_index,
    vals_index,
    options_a_template_id,
    template,
    permutation_index
)
```

### Important Caveat

Even with matching keys, **the lotteries are fundamentally different** between Treatment and Control:
- Treatment has a certain option
- Control has two risky options
- Expected values differ

This is **not** the classic Kahneman design where Treatment and Control have identical expected values. The current design tests whether the presence of certainty changes preferences, not whether "certainty framing" of the same EV changes preferences.

## Analysis Output Files

When running `run_analysis.py`, the following files are generated:

| File | Contents |
|------|----------|
| `_[...]_.csv` | Summary: % choosing target vs better_expected_value per condition |
| `_[...]_confidences.csv` | 95% bootstrapped confidence intervals |
| `_[...]_full_answers.csv` | Per-sample answers (wide format) |
| `logging_aux_[...].txt` | Detailed breakdown by template and position |

### Full Answers CSV Structure

4 rows × N columns (one column per sample):

| Row | Contents |
|-----|----------|
| `Target is prize with certainty` | Treatment answers (`target` or `better_expected_value`) |
| `Probabilities Target is prize with certainty` | Treatment log-probabilities |
| `Target is risky too` | Control answers |
| `Probabilities Target is risky too` | Control log-probabilities |

## Bias Score Calculation

```
Bias Score = P(choose target | Treatment) - P(choose target | Control)
```

For OLMo-3-7B-Instruct: 59.2% - 33.7% = **25.5% certainty effect**

### Position Bias Control

Position bias is controlled through balanced `permutation_index`:
- ~50% of samples have target in position 1
- ~50% of samples have target in position 2
- Both Treatment and Control have the same position distribution
- Position effects cancel in the subtraction

## Statistical Power

For a 25% effect size:

| Power | Samples per group |
|-------|-------------------|
| 80% | ~57 |
| 90% | ~76 |
| 95% | ~93 |

The current dataset (504 Treatment, 336 Control) has >99.9% power to detect this effect.

## Recommendations

1. **For cleaner paired analysis**, filter to `bias_type_index=1` (Common Consequence Effect)
2. **For true Kahneman-style pairing**, the data generation would need modification to ensure Treatment B and Control B have identical expected values
3. **Position bias is substantial** in some models - always check per-position breakdowns in the logging output

## Code References

- Data generation: `Data_generation/generate_samples_certainty.py`
- Value definitions: `Data_generation/certainty_values.py`
- Templates: `Data_generation/templates.py` (see `CERTAINTY_TEMPLATES`)
- Analysis: `Analysis/certainty_analysis.py`
- Answer parsing: `Analysis/analyze.py:find_ans_in_tokens()`
