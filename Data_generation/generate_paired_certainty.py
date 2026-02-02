"""
Generate paired certainty effect questions.

In paired design:
- Option A (better expected value) is IDENTICAL between treatment and control
- Option B (target) has the SAME expected value, but:
  - Treatment: 100% certain prize
  - Control: Risky with same expected value (e.g., 80% to win higher amount, 20% $0)

This isolates the certainty manipulation as the only difference between conditions.
"""

import argparse
import itertools
import copy
import json
from pathlib import Path
from string import Template

from templates import CERTAINTY_TEMPLATES
from samples_classes import Certainty_type


def get_paired_values():
    """
    Generate paired values where:
    - Option A (risky, higher EV) is the same for treatment and control
    - Option B (target) has same EV but certain (treatment) vs risky (control)

    Returns list of dicts with all the values needed for both conditions.
    """
    paired_values = [
        # Pair 1: Option A = 80% $4000 (EV=3200), Option B EV = $3000
        {
            "option_a": {"prob1": 80, "prize1": 4000, "prob2": 20, "prize2": 0},
            "option_b_treatment": {"prob1": 100, "prize1": 3000},  # Certain $3000
            "option_b_control": {"prob1": 75, "prize1": 4000, "prob2": 25, "prize2": 0},  # EV = $3000
        },
        # Pair 2: Option A = 80% $5000 (EV=4000), Option B EV = $3000
        {
            "option_a": {"prob1": 80, "prize1": 5000, "prob2": 20, "prize2": 0},
            "option_b_treatment": {"prob1": 100, "prize1": 3000},  # Certain $3000
            "option_b_control": {"prob1": 60, "prize1": 5000, "prob2": 40, "prize2": 0},  # EV = $3000
        },
        # Pair 3: Option A = 85% $5000 (EV=4250), Option B EV = $4000
        {
            "option_a": {"prob1": 85, "prize1": 5000, "prob2": 15, "prize2": 0},
            "option_b_treatment": {"prob1": 100, "prize1": 4000},  # Certain $4000
            "option_b_control": {"prob1": 80, "prize1": 5000, "prob2": 20, "prize2": 0},  # EV = $4000
        },
        # Pair 4: Option A = 85% $6000 (EV=5100), Option B EV = $5000
        {
            "option_a": {"prob1": 85, "prize1": 6000, "prob2": 15, "prize2": 0},
            "option_b_treatment": {"prob1": 100, "prize1": 5000},  # Certain $5000
            "option_b_control": {"prob1": 83, "prize1": 6000, "prob2": 17, "prize2": 0},  # EV ≈ $5000
        },
        # Pair 5: Option A = 90% $3000 (EV=2700), Option B EV = $2000
        {
            "option_a": {"prob1": 90, "prize1": 3000, "prob2": 10, "prize2": 0},
            "option_b_treatment": {"prob1": 100, "prize1": 2000},  # Certain $2000
            "option_b_control": {"prob1": 67, "prize1": 3000, "prob2": 33, "prize2": 0},  # EV ≈ $2000
        },
        # Pair 6: Option A = 70% $3000 (EV=2100), Option B EV = $2000
        {
            "option_a": {"prob1": 70, "prize1": 3000, "prob2": 30, "prize2": 0},
            "option_b_treatment": {"prob1": 100, "prize1": 2000},  # Certain $2000
            "option_b_control": {"prob1": 67, "prize1": 3000, "prob2": 33, "prize2": 0},  # EV ≈ $2000
        },
        # Pair 7: Option A = 60% $2000 (EV=1200), Option B EV = $1000
        {
            "option_a": {"prob1": 60, "prize1": 2000, "prob2": 40, "prize2": 0},
            "option_b_treatment": {"prob1": 100, "prize1": 1000},  # Certain $1000
            "option_b_control": {"prob1": 50, "prize1": 2000, "prob2": 50, "prize2": 0},  # EV = $1000
        },
    ]
    return paired_values


def build_option_text(option_vals, is_certain):
    """Build the option text from values."""
    if is_certain:
        return f"${option_vals['prize1']} with certainty"
    else:
        return f"{option_vals['prob1']}% to win ${option_vals['prize1']}, {option_vals['prob2']}% to win ${option_vals['prize2']}"


def build_option_dict(option_vals, option_type, is_certain):
    """Build option dictionary with text and metadata."""
    option = {
        "PROB1": f"{option_vals['prob1']}%",
        "PRIZE1": f"${option_vals['prize1']}",
        "option_type": option_type,
    }
    if not is_certain:
        option["PROB2"] = f"{option_vals['prob2']}%"
        option["PRIZE2"] = f"${option_vals['prize2']}"
    option["option_text"] = build_option_text(option_vals, is_certain)
    return option


def generate_sample_text(template_id, first_option_opening, second_option_opening,
                         option_a_text, option_b_text):
    """Generate the full question text from template."""
    template = CERTAINTY_TEMPLATES["CERTAINTY_BIAS_MEGA"][template_id]
    text = template.safe_substitute(
        FIRST_OPTION_OPENING=first_option_opening,
        SECOND_OPTION_OPENING=second_option_opening,
        FIRST_OPTION=option_a_text,
        SECOND_OPTION=option_b_text,
    )
    return text


def generate_paired_samples(args):
    """Generate all paired treatment/control samples."""
    paired_values = get_paired_values()

    # Get all template combinations
    all_options_text = list(CERTAINTY_TEMPLATES["ALL_OPTIONS_TEXT_CERTAINTY"].items())

    treatment_samples = []
    control_samples = []

    sample_id = 0

    for pair_id, pair in enumerate(paired_values):
        for template_id in args.templates:
            for options_text_id, (first_opt, second_opt) in all_options_text:
                # Build Option A (same for both conditions)
                option_a = build_option_dict(pair["option_a"], "better_expected_value", is_certain=False)

                # Build Option B for treatment (certain)
                option_b_treatment = build_option_dict(
                    pair["option_b_treatment"], "target", is_certain=True
                )

                # Build Option B for control (risky, same EV)
                option_b_control = build_option_dict(
                    pair["option_b_control"], "target", is_certain=False
                )

                # Generate permutations (A first vs B first)
                permutations = [(option_a, option_b_treatment, option_a, option_b_control)]
                if args.all_options_permutations:
                    permutations.append((option_b_treatment, option_a, option_b_control, option_a))

                for perm_idx, (opt_a_t, opt_b_t, opt_a_c, opt_b_c) in enumerate(permutations):
                    # Determine answer indices based on permutation
                    if perm_idx == 0:
                        # Option A first: better_expected_value=1, target=2
                        better_ev_idx = 1
                        target_idx = 2
                    else:
                        # Option B first: target=1, better_expected_value=2
                        better_ev_idx = 2
                        target_idx = 1

                    metadata_base = {
                        "pair_id": pair_id,
                        "template": template_id,
                        "options_text_id": options_text_id,
                        "permutation_index": perm_idx + 1,
                        "first_option_opening": first_opt,
                        "second_option_opening": second_opt,
                        "option_a": opt_a_t if perm_idx == 0 else opt_b_t,
                        "option_b": opt_b_t if perm_idx == 0 else opt_a_t,
                        "better_expected_value": better_ev_idx,
                        "target": target_idx,
                        "bias_type": "paired_certainty",
                        "bias_name": "certainty",
                        "option_a_ev": pair["option_a"]["prob1"] * pair["option_a"]["prize1"] / 100,
                        "option_b_ev": pair["option_b_treatment"]["prize1"],  # Same EV for both conditions
                    }

                    # Treatment sample
                    treatment_text = generate_sample_text(
                        template_id, first_opt, second_opt,
                        opt_a_t["option_text"], opt_b_t["option_text"]
                    )
                    treatment_metadata = copy.deepcopy(metadata_base)
                    treatment_metadata["human_or_right_answer"] = target_idx  # Humans prefer certain
                    treatment_metadata["with_bias"] = True
                    treatment_metadata["text"] = treatment_text
                    treatment_metadata["option_b_certain"] = True

                    treatment_samples.append({
                        "sample_id": sample_id,
                        "pair_id": pair_id,
                        "text": treatment_text,
                        "metadata": treatment_metadata,
                    })

                    # Control sample (update option_b to risky version)
                    control_metadata = copy.deepcopy(metadata_base)
                    control_metadata["option_a"] = opt_a_c if perm_idx == 0 else opt_b_c
                    control_metadata["option_b"] = opt_b_c if perm_idx == 0 else opt_a_c
                    control_metadata["human_or_right_answer"] = better_ev_idx  # Should prefer higher EV when both risky
                    control_metadata["with_bias"] = False
                    control_metadata["option_b_certain"] = False

                    control_text = generate_sample_text(
                        template_id, first_opt, second_opt,
                        opt_a_c["option_text"], opt_b_c["option_text"]
                    )
                    control_metadata["text"] = control_text

                    control_samples.append({
                        "sample_id": sample_id,
                        "pair_id": pair_id,
                        "text": control_text,
                        "metadata": control_metadata,
                    })

                    sample_id += 1

    return treatment_samples, control_samples


def save_samples(samples, output_path):
    """Save samples to JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert to dict format expected by the prediction pipeline
    samples_dict = {str(i): sample for i, sample in enumerate(samples)}

    with open(output_path, "w") as f:
        json.dump(samples_dict, f, indent=2)

    print(f"Saved {len(samples)} samples to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate paired certainty effect samples")
    parser.add_argument(
        "--templates",
        type=int,
        nargs="+",
        default=[1, 2, 3],
        help="Template IDs to use",
    )
    parser.add_argument(
        "--all_options_permutations",
        action="store_true",
        default=True,
        help="Generate all permutations of option ordering",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="Data/certainty_paired/all_permutations",
        help="Output directory for generated samples",
    )
    parser.add_argument(
        "--bias_types",
        type=str,
        default="paired",
        help="Bias type identifier for filenames",
    )

    args = parser.parse_args()

    treatment_samples, control_samples = generate_paired_samples(args)

    output_dir = Path(args.output_dir)
    templates_str = str(args.templates).replace(" ", "")

    treatment_path = output_dir / f"t_{templates_str}_{args.bias_types}_Treatment.json"
    control_path = output_dir / f"t_{templates_str}_{args.bias_types}_Control.json"

    save_samples(treatment_samples, treatment_path)
    save_samples(control_samples, control_path)

    # Print summary
    print(f"\nGenerated {len(treatment_samples)} treatment samples and {len(control_samples)} control samples")
    print(f"Number of unique pairs: {len(get_paired_values())}")

    # Print example pair
    print("\n--- Example Pair ---")
    print(f"Treatment: {treatment_samples[0]['text'][:200]}...")
    print(f"Control:   {control_samples[0]['text'][:200]}...")


if __name__ == "__main__":
    main()
