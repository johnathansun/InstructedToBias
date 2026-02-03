# =============================================================================
# CELL 1: Evaluation functions
# =============================================================================

def evaluate_steering_on_test(test_df, layer, steering_vec, strength):
    """
    Evaluate steering effectiveness on test set.
    Returns a DataFrame with all responses (original and steered) plus a flip rate.
    """
    results = []

    for idx, row in tqdm(test_df.iterrows(), total=len(test_df)):
        treatment_text = row['treatment_text']
        control_text = row['control_text']
        original_treatment_resp = row['treatment_resp']
        original_control_resp = row['control_resp']

        # Apply steering (negative strength to steer away from certainty bias)
        steered_resp_full = get_steered_response(treatment_text, layer, steering_vec, strength)
        steered_resp_parsed = parse_answer(steered_resp_full)

        # Check if it flipped to control's answer
        flipped = (steered_resp_parsed == original_control_resp and
                   original_treatment_resp != original_control_resp)

        results.append({
            'index': idx,
            'control_text': control_text,
            'treatment_text': treatment_text,
            'original_control_resp': original_control_resp,
            'original_treatment_resp': original_treatment_resp,
            'steered_resp_full': steered_resp_full,
            'steered_resp_parsed': steered_resp_parsed,
            'layer': layer,
            'strength': strength,
            'flipped_to_control': flipped
        })

    results_df = pd.DataFrame(results)
    flip_rate = results_df['flipped_to_control'].mean()

    return results_df, flip_rate


def run_steering_evaluation(test_df, layers_to_test, steering_vectors, strengths, save_path=None):
    """
    Run full steering evaluation across multiple layers and strengths.
    """
    all_results = []

    for layer in layers_to_test:
        for strength in strengths:
            print(f"Evaluating layer {layer}, strength {strength}...")
            results_df, flip_rate = evaluate_steering_on_test(
                test_df, layer, steering_vectors[layer], strength
            )
            all_results.append(results_df)
            print(f"  Flip rate: {flip_rate:.1%}")

    combined_df = pd.concat(all_results, ignore_index=True)

    if save_path:
        combined_df.to_csv(save_path, index=False)
        print(f"\nSaved results to {save_path}")

    return combined_df


# =============================================================================
# CELL 2: Run the evaluation
# =============================================================================

print("Evaluating steering vectors on test set...")
steering_results = run_steering_evaluation(
    test_df=df_switch_test,
    layers_to_test=top_layers[:3].tolist(),
    steering_vectors=steering_vectors,
    strengths=[-0.5, -1.0, -2.0],
    save_path='steering_evaluation_results.csv'
)

# Summary by layer and strength
print("\nSummary (flip rate by layer and strength):")
print(steering_results.groupby(['layer', 'strength'])['flipped_to_control'].mean())
