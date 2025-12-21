"""
Calculate directional accuracy for SA backtest results.

Directional accuracy measures how often the predicted direction (sign)
of the change matches the actual direction.
"""

import pandas as pd
import numpy as np

# Read the backtest results
df = pd.read_csv('_output/backtest_results/sa/backtest_results_sa.csv')

print("=" * 80)
print("DIRECTIONAL ACCURACY ANALYSIS - SA (Seasonally Adjusted)")
print("=" * 80)
print(f"\nTotal number of predictions: {len(df)}")

# Remove any rows with missing values in actual or predicted
df_clean = df.dropna(subset=['actual', 'predicted'])
print(f"Valid predictions (non-null): {len(df_clean)}")

if len(df_clean) == 0:
    print("\nNo valid data to calculate directional accuracy!")
    exit(1)

# Calculate directional accuracy
# We check if the sign (positive/negative) of actual and predicted match
actual_direction = np.sign(df_clean['actual'])
predicted_direction = np.sign(df_clean['predicted'])

# Count matches
matches = (actual_direction == predicted_direction).sum()
directional_accuracy = matches / len(df_clean) * 100

print(f"\n{'=' * 80}")
print("RESULTS")
print(f"{'=' * 80}")
print(f"Correct directional predictions: {matches}/{len(df_clean)}")
print(f"Directional Accuracy: {directional_accuracy:.2f}%")

# Additional breakdown by direction
positive_actual = df_clean[df_clean['actual'] > 0]
negative_actual = df_clean[df_clean['actual'] < 0]
zero_actual = df_clean[df_clean['actual'] == 0]

if len(positive_actual) > 0:
    positive_correct = ((positive_actual['predicted'] > 0).sum())
    positive_accuracy = positive_correct / len(positive_actual) * 100
    print(f"\nAccuracy when actual > 0: {positive_correct}/{len(positive_actual)} ({positive_accuracy:.2f}%)")

if len(negative_actual) > 0:
    negative_correct = ((negative_actual['predicted'] < 0).sum())
    negative_accuracy = negative_correct / len(negative_actual) * 100
    print(f"Accuracy when actual < 0: {negative_correct}/{len(negative_actual)} ({negative_accuracy:.2f}%)")

if len(zero_actual) > 0:
    zero_correct = ((zero_actual['predicted'] == 0).sum())
    print(f"Accuracy when actual = 0: {zero_correct}/{len(zero_actual)}")

# Show some sample predictions
print(f"\n{'=' * 80}")
print("SAMPLE PREDICTIONS (First 10)")
print(f"{'=' * 80}")
print(f"{'Date':<12} {'Actual':>10} {'Predicted':>10} {'Actual Dir':>12} {'Pred Dir':>12} {'Match':>8}")
print("-" * 80)

for idx, row in df_clean.head(10).iterrows():
    actual = row['actual']
    predicted = row['predicted']
    actual_dir = 'UP' if actual > 0 else ('DOWN' if actual < 0 else 'ZERO')
    pred_dir = 'UP' if predicted > 0 else ('DOWN' if predicted < 0 else 'ZERO')
    match = '✓' if np.sign(actual) == np.sign(predicted) else '✗'

    print(f"{row['ds']:<12} {actual:>10.0f} {predicted:>10.1f} {actual_dir:>12} {pred_dir:>12} {match:>8}")

# Show mismatches
mismatches = df_clean[actual_direction != predicted_direction]
if len(mismatches) > 0:
    print(f"\n{'=' * 80}")
    print(f"DIRECTIONAL MISMATCHES ({len(mismatches)} total)")
    print(f"{'=' * 80}")
    print(f"{'Date':<12} {'Actual':>10} {'Predicted':>10} {'Actual Dir':>12} {'Pred Dir':>12}")
    print("-" * 80)
    for idx, row in mismatches.iterrows():
        actual = row['actual']
        predicted = row['predicted']
        actual_dir = 'UP' if actual > 0 else ('DOWN' if actual < 0 else 'ZERO')
        pred_dir = 'UP' if predicted > 0 else ('DOWN' if predicted < 0 else 'ZERO')
        print(f"{row['ds']:<12} {actual:>10.0f} {predicted:>10.1f} {actual_dir:>12} {pred_dir:>12}")

print("=" * 80)

# Save detailed results to a file
output_data = df_clean.copy()
output_data['actual_direction'] = actual_direction.apply(lambda x: 'UP' if x > 0 else ('DOWN' if x < 0 else 'ZERO'))
output_data['predicted_direction'] = predicted_direction.apply(lambda x: 'UP' if x > 0 else ('DOWN' if x < 0 else 'ZERO'))
output_data['direction_match'] = (actual_direction == predicted_direction)

output_file = '_output/backtest_results/sa/directional_accuracy_analysis.csv'
output_data.to_csv(output_file, index=False)
print(f"\nDetailed analysis saved to: {output_file}")
