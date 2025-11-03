"""
Question 3: Experiment with n_estimators Parameter

Try different values of n_estimators from 10 to 200 with step 10.
Set random_state to 1.
Evaluate the model on the validation dataset.

After which value of n_estimators does RMSE stop improving?
Consider 3 decimal places for calculating the answer.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("QUESTION 3: EXPERIMENTING WITH n_estimators PARAMETER")
print("="*80)

# ==============================================================================
# STEP 1: LOAD AND PREPARE DATA
# ==============================================================================
print("\n📝 Step 1: Loading and preparing data...")

df = pd.read_csv('car_fuel_efficiency.csv')
df_prep = df.copy()
df_prep = df_prep.fillna(0)

y = df_prep['fuel_efficiency_mpg'].values
X_full = df_prep.drop('fuel_efficiency_mpg', axis=1)

# Train/validation/test split (60/20/20)
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X_full, y, test_size=0.2, random_state=1
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full, test_size=0.25, random_state=1
)

print(f"   ✓ Train set: {len(X_train)} samples (60%)")
print(f"   ✓ Validation set: {len(X_val)} samples (20%)")
print(f"   ✓ Test set: {len(X_test)} samples (20%)")

# Use DictVectorizer
dv = DictVectorizer(sparse=True)
train_dict = X_train.to_dict(orient='records')
X_train = dv.fit_transform(train_dict)

val_dict = X_val.to_dict(orient='records')
X_val = dv.transform(val_dict)

print(f"   ✓ Feature matrix shape: {X_train.shape}")

# ==============================================================================
# STEP 2: EXPERIMENT WITH n_estimators
# ==============================================================================
print("\n🌲 Step 2: Training Random Forest models with different n_estimators...")
print("   Testing n_estimators from 10 to 200 (step 10)")

n_estimators_values = list(range(10, 201, 10))
results = []

print("\n   Progress:")
print("   " + "-" * 70)

for n_est in n_estimators_values:
    rf = RandomForestRegressor(
        n_estimators=n_est,
        random_state=1,
        n_jobs=-1
    )
    
    rf.fit(X_train, y_train)
    
    # Predictions
    y_val_pred = rf.predict(X_val)
    
    # Calculate RMSE
    rmse_val = np.sqrt(mean_squared_error(y_val, y_val_pred))
    
    results.append({
        'n_estimators': n_est,
        'rmse': rmse_val
    })
    
    # Print progress every 10 models
    if n_est % 20 == 0 or n_est == 10 or n_est == 200:
        print(f"     n_estimators={n_est:3d} → RMSE: {rmse_val:.6f}")

print("\n   ✓ Training complete!")

# Convert to DataFrame for easier analysis
results_df = pd.DataFrame(results)
results_df['rmse_rounded'] = results_df['rmse'].round(3)

# ==============================================================================
# STEP 3: FIND WHEN RMSE STOPS IMPROVING
# ==============================================================================
print("\n🔍 Step 3: Finding when RMSE stops improving (3 decimal places)...")
print("   " + "-" * 70)

# Round to 3 decimal places for comparison
results_df['rmse_3dp'] = results_df['rmse'].round(3)

# Find when RMSE stops improving (when rounded to 3 decimal places)
# "Stops improving" means: after this value, RMSE (rounded to 3dp) never gets better

# Find the minimum RMSE value (rounded to 3 decimal places)
min_rmse_3dp = results_df['rmse_3dp'].min()
min_rmse_rows = results_df[results_df['rmse_3dp'] == min_rmse_3dp]
first_min_n_est = min_rmse_rows.iloc[0]['n_estimators']

print(f"\n   Minimum RMSE (rounded to 3 decimal places): {min_rmse_3dp:.3f}")
print(f"   First achieved at n_estimators: {int(first_min_n_est)}")

# Find the last value where RMSE actually improves (decreases) at 3 decimal places
# "Stops improving" = last value where we see an improvement, after which it never improves again
answer = None
for i in range(len(results_df) - 1):
    current_rmse_3dp = results_df.iloc[i]['rmse_3dp']
    next_rmse_3dp = results_df.iloc[i + 1]['rmse_3dp']
    
    if next_rmse_3dp < current_rmse_3dp:
        # There's an improvement, so update the answer
        answer = int(results_df.iloc[i + 1]['n_estimators'])
        print(f"   Found improvement: n_estimators={results_df.iloc[i]['n_estimators']} → {results_df.iloc[i+1]['n_estimators']}: {current_rmse_3dp:.3f} → {next_rmse_3dp:.3f}")

# If no improvement found after the minimum, use the first minimum
if answer is None:
    answer = int(first_min_n_est)

# Check if RMSE continues to improve after 'answer'
continue_improving = False
answer_idx = results_df[results_df['n_estimators'] == answer].index[0]
for idx in range(answer_idx + 1, len(results_df)):
    if results_df.iloc[idx]['rmse_3dp'] < results_df.iloc[answer_idx]['rmse_3dp']:
        continue_improving = True
        break

if not continue_improving:
    print(f"\n   RMSE (rounded to 3dp) stops improving after n_estimators={answer}")
else:
    # If it continues improving, find the actual last improvement
    print(f"\n   Checking for improvements after n_estimators={answer}...")

# ==============================================================================
# STEP 4: DETAILED ANALYSIS
# ==============================================================================
print("\n" + "="*80)
print("📊 DETAILED RESULTS")
print("="*80)

print("\n   RMSE values (rounded to 3 decimal places):")
print("   " + "-" * 70)
print(f"   {'n_estimators':>12} | {'RMSE (full)':>12} | {'RMSE (3dp)':>12} | Improvement")
print("   " + "-" * 70)

previous_rmse_3dp = None
for idx, row in results_df.iterrows():
    current_rmse = row['rmse']
    current_rmse_3dp = row['rmse_3dp']
    
    if previous_rmse_3dp is not None:
        improvement = previous_rmse_3dp - current_rmse_3dp
        if improvement > 0:
            improvement_str = f"↓ {improvement:.3f}"
        elif improvement < 0:
            improvement_str = f"↑ {abs(improvement):.3f}"
        else:
            improvement_str = "→ 0.000 (stopped improving)"
    else:
        improvement_str = "-"
    
    marker = " ⭐" if row['n_estimators'] == answer else ""
    print(f"   {row['n_estimators']:>12} | {current_rmse:>12.6f} | {current_rmse_3dp:>12.3f} | {improvement_str:>20}{marker}")
    
    previous_rmse_3dp = current_rmse_3dp

# ==============================================================================
# STEP 5: COMPARE WITH ANSWER OPTIONS
# ==============================================================================
print("\n" + "="*80)
print("🎯 ANSWER ANALYSIS")
print("="*80)

options = [10, 25, 80, 200]

print(f"\n   RMSE stops improving (at 3 decimal places) at: n_estimators = {answer}")
print(f"\n   Multiple choice options:")
print("   " + "-" * 70)

for opt in options:
    match = "✅" if opt == answer else "  "
    if opt in results_df['n_estimators'].values:
        rmse_at_opt = results_df[results_df['n_estimators'] == opt]['rmse'].values[0]
        rmse_3dp_at_opt = results_df[results_df['n_estimators'] == opt]['rmse_3dp'].values[0]
        print(f"   {match} {opt} → RMSE: {rmse_at_opt:.6f} ({rmse_3dp_at_opt:.3f})")
    else:
        # Find closest value
        closest = results_df.iloc[(results_df['n_estimators'] - opt).abs().argsort()[:1]]
        closest_n_est = int(closest['n_estimators'].values[0])
        rmse_at_closest = closest['rmse'].values[0]
        rmse_3dp_at_closest = closest['rmse_3dp'].values[0]
        print(f"   {match} {opt} (closest: {closest_n_est}) → RMSE: {rmse_at_closest:.6f} ({rmse_3dp_at_closest:.3f})")

print(f"\n   🎯 Answer: {answer}")

# Show when improvements become negligible
print("\n   Additional insight:")
print("   " + "-" * 70)
improvements = []
for i in range(1, len(results_df)):
    prev_rmse = results_df.iloc[i-1]['rmse']
    curr_rmse = results_df.iloc[i]['rmse']
    improvement = prev_rmse - curr_rmse
    improvements.append(improvement)
    
    if improvement < 0.0001:  # Less than 0.0001 improvement
        print(f"   At n_estimators={results_df.iloc[i]['n_estimators']}: improvement = {improvement:.6f} (< 0.0001)")

# ==============================================================================
# STEP 6: VISUALIZATION
# ==============================================================================
print("\n📈 Step 4: Creating visualization...")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 1. RMSE vs n_estimators (full precision)
ax1 = axes[0, 0]
ax1.plot(results_df['n_estimators'], results_df['rmse'], 
         marker='o', linewidth=2, markersize=4, color='steelblue')
ax1.axvline(x=answer, color='red', linestyle='--', linewidth=2, 
            label=f'Stops improving: {answer}')
ax1.set_xlabel('n_estimators', fontsize=12, fontweight='bold')
ax1.set_ylabel('RMSE (Validation)', fontsize=12, fontweight='bold')
ax1.set_title('RMSE vs n_estimators (Full Precision)', 
              fontsize=14, fontweight='bold')
ax1.grid(alpha=0.3)
ax1.legend()

# 2. RMSE vs n_estimators (rounded to 3 decimal places)
ax2 = axes[0, 1]
ax2.plot(results_df['n_estimators'], results_df['rmse_3dp'], 
         marker='o', linewidth=2, markersize=4, color='green')
ax2.axvline(x=answer, color='red', linestyle='--', linewidth=2, 
            label=f'Stops improving: {answer}')
ax2.set_xlabel('n_estimators', fontsize=12, fontweight='bold')
ax2.set_ylabel('RMSE (3 decimal places)', fontsize=12, fontweight='bold')
ax2.set_title('RMSE vs n_estimators (Rounded to 3dp)\nStops improving at {}'.format(answer), 
              fontsize=14, fontweight='bold')
ax2.grid(alpha=0.3)
ax2.legend()

# 3. Improvement between consecutive values
ax3 = axes[1, 0]
improvements = []
for i in range(1, len(results_df)):
    improvement = results_df.iloc[i-1]['rmse_3dp'] - results_df.iloc[i]['rmse_3dp']
    improvements.append(improvement)

n_est_mid = [(results_df.iloc[i-1]['n_estimators'] + results_df.iloc[i]['n_estimators']) / 2 
             for i in range(1, len(results_df))]

colors_imp = ['green' if imp > 0 else 'red' if imp < 0 else 'gray' for imp in improvements]
bars = ax3.bar(range(len(improvements)), improvements, color=colors_imp, alpha=0.7)
ax3.axhline(y=0, color='black', linestyle='-', linewidth=1)
ax3.set_xlabel('n_estimators (between values)', fontsize=12, fontweight='bold')
ax3.set_ylabel('RMSE Improvement (3dp)', fontsize=12, fontweight='bold')
ax3.set_title('Improvement Between Consecutive n_estimators\n(Green = Improving, Red = Worse, Gray = No Change)', 
              fontsize=14, fontweight='bold')
ax3.set_xticks(range(0, len(improvements), 5))
ax3.set_xticklabels([int(n_est_mid[i]) for i in range(0, len(improvements), 5)], rotation=45)
ax3.grid(axis='y', alpha=0.3)

# 4. Zoomed view around the answer
ax4 = axes[1, 1]
# Show around the answer value
answer_idx = results_df[results_df['n_estimators'] == answer].index[0]
start_idx = max(0, answer_idx - 5)
end_idx = min(len(results_df), answer_idx + 6)
zoom_df = results_df.iloc[start_idx:end_idx]

ax4.plot(zoom_df['n_estimators'], zoom_df['rmse_3dp'], 
         marker='o', linewidth=2, markersize=8, color='steelblue')
ax4.axvline(x=answer, color='red', linestyle='--', linewidth=2, 
            label=f'Stops improving: {answer}')
ax4.scatter([answer], [results_df[results_df['n_estimators'] == answer]['rmse_3dp'].values[0]], 
            color='red', s=200, zorder=5, marker='*', label='Answer')
ax4.set_xlabel('n_estimators', fontsize=12, fontweight='bold')
ax4.set_ylabel('RMSE (3 decimal places)', fontsize=12, fontweight='bold')
ax4.set_title(f'Zoomed View Around Answer (n_estimators={answer})', 
              fontsize=14, fontweight='bold')
ax4.grid(alpha=0.3)
ax4.legend()

plt.tight_layout()
plt.savefig('question3_n_estimators_analysis.png', dpi=300, bbox_inches='tight')
print("   ✓ Visualization saved as 'question3_n_estimators_analysis.png'")

# ==============================================================================
# SUMMARY
# ==============================================================================
print("\n" + "="*80)
print("✨ SUMMARY")
print("="*80)

print(f"\n   Question: After which value of n_estimators does RMSE stop improving?")
print(f"   (Consider 3 decimal places)")
print(f"\n   ✅ Answer: {answer}")
print(f"\n   Explanation:")
print(f"      • Tested n_estimators from 10 to 200 (step 10)")
print(f"      • RMSE values rounded to 3 decimal places")
print(f"      • At n_estimators={answer}, RMSE reaches its minimum value of {min_rmse_3dp:.3f}")
print(f"      • After this point, RMSE (rounded to 3dp) does not improve further")

# Show the exact values around the answer
answer_row = results_df[results_df['n_estimators'] == answer].iloc[0]
print(f"\n   Details at n_estimators={answer}:")
print(f"      • RMSE (full): {answer_row['rmse']:.6f}")
print(f"      • RMSE (3dp): {answer_row['rmse_3dp']:.3f}")

plt.show()

print("\n" + "="*80)
print("✨ Analysis Complete!")
print("="*80)

