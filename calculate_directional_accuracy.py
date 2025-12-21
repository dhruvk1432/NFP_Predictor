import pandas as pd
import numpy as np
import os

# Read the backtest results
file_path = '_output/backtest_results/sa/backtest_results_sa.csv'
if not os.path.exists(file_path):
    print(f"Error: File not found at {file_path}")
    exit(1)

df = pd.read_csv(file_path)

print("=" * 80)
print("DIRECTIONAL ACCURACY ANALYSIS")
print("=" * 80)
print(f"\nTotal rows in file: {len(df)}")

# 1. Calculate differences FIRST on the full dataframe
df["actual_diff"] = df["actual"].diff()
df["predicted_diff"] = df["predicted"].diff()

# 2. Drop rows where we can't calculate direction (the first row) 
# and rows with missing actual/predicted values
df_clean = df.dropna(subset=['actual_diff', 'predicted_diff', 'actual', 'predicted']).copy()

print(f"Valid prediction pairs (after diff): {len(df_clean)}")

if len(df_clean) == 0:
    print("\nNo valid data to calculate directional accuracy!")
    exit(1)

# 3. Calculate directional accuracy
# np.sign returns -1, 0, or 1
actual_direction = np.sign(df_clean['actual_diff'])
predicted_direction = np.sign(df_clean['predicted_diff'])

# Count matches
matches = (actual_direction == predicted_direction).sum()
directional_accuracy = (matches / len(df_clean)) * 100

print(f"\n{'=' * 80}")
print("RESULTS")
print(f"{'=' * 80}")
print(f"Correct directional predictions: {matches}/{len(df_clean)}")
print(f"Directional Accuracy: {directional_accuracy:.2f}%")

# Additional breakdown by direction
positive_actual = df_clean[df_clean['actual_diff'] > 0]
negative_actual = df_clean[df_clean['actual_diff'] < 0]
zero_actual = df_clean[df_clean['actual_diff'] == 0]

if len(positive_actual) > 0:
    positive_correct = (positive_actual['predicted_diff'] > 0).sum()
    print(f"\nAccuracy when actual > 0: {positive_correct}/{len(positive_actual)} ({positive_correct/len(positive_actual)*100:.2f}%)")

if len(negative_actual) > 0:
    negative_correct = (negative_actual['predicted_diff'] < 0).sum()
    print(f"Accuracy when actual < 0: {negative_correct}/{len(negative_actual)} ({negative_correct/len(negative_actual)*100:.2f}%)")

if len(zero_actual) > 0:
    zero_correct = (zero_actual['predicted_diff'] == 0).sum()
    print(f"Accuracy when actual = 0: {zero_correct}/{len(zero_actual)}")

# Show some sample predictions
print(f"\n{'=' * 80}")
print("SAMPLE PREDICTIONS (First 10)")
print(f"{'=' * 80}")
print(f"{'Date':<12} {'Actual_Δ':>10} {'Pred_Δ':>10} {'Actual Dir':>12} {'Pred Dir':>12} {'Match':>8}")
print("-" * 80)

for idx, row in df_clean.head(10).iterrows():
    act_diff = row['actual_diff']
    pred_diff = row['predicted_diff']
    actual_dir = 'UP' if act_diff > 0 else ('DOWN' if act_diff < 0 else 'ZERO')
    pred_dir = 'UP' if pred_diff > 0 else ('DOWN' if pred_diff < 0 else 'ZERO')
    match = '✓' if np.sign(act_diff) == np.sign(pred_diff) else '✗'

    # Using 'ds' if it exists, otherwise index
    date_val = row['ds'] if 'ds' in row else str(idx)
    print(f"{date_val:<12} {act_diff:>10.2f} {pred_diff:>10.2f} {actual_dir:>12} {pred_dir:>12} {match:>8}")

print("=" * 80)

# Save detailed results
df_clean['actual_direction_label'] = actual_direction.map({1: 'UP', -1: 'DOWN', 0: 'ZERO'})
df_clean['predicted_direction_label'] = predicted_direction.map({1: 'UP', -1: 'DOWN', 0: 'ZERO'})
df_clean['direction_match'] = (actual_direction == predicted_direction)

output_file = '_output/backtest_results/nsa/directional_accuracy_analysis.csv'
# Ensure directory exists
os.makedirs(os.path.dirname(output_file), exist_ok=True)
df_clean.to_csv(output_file, index=False)
print(f"\nDetailed analysis saved to: {output_file}")
