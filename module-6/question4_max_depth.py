"""
Question 4: Select the best max_depth

Try different values of max_depth: [10, 15, 20, 25]
For each of these values,
  - try different values of n_estimators from 10 till 200 (with step 10)
  - calculate the mean RMSE
Fix the random seed: random_state=1

What's the best max_depth, using the mean RMSE?
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("QUESTION 4: SELECTING THE BEST max_depth")
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
# STEP 2: EXPERIMENT WITH max_depth AND n_estimators
# ==============================================================================
print("\n🌲 Step 2: Training Random Forest models with different max_depth and n_estimators...")
print("   Testing max_depth values: [10, 15, 20, 25]")
print("   For each max_depth, testing n_estimators from 10 to 200 (step 10)")

max_depth_values = [10, 15, 20, 25]
n_estimators_values = list(range(10, 201, 10))

results = []

print("\n   Progress:")
print("   " + "-" * 80)

for max_d in max_depth_values:
    rmse_values_for_depth = []
    
    print(f"\n   Testing max_depth={max_d}:")
    
    for n_est in n_estimators_values:
        rf = RandomForestRegressor(
            n_estimators=n_est,
            max_depth=max_d,
            random_state=1,
            n_jobs=-1
        )
        
        rf.fit(X_train, y_train)
        
        # Predictions
        y_val_pred = rf.predict(X_val)
        
        # Calculate RMSE
        rmse_val = np.sqrt(mean_squared_error(y_val, y_val_pred))
        
        rmse_values_for_depth.append(rmse_val)
        
        results.append({
            'max_depth': max_d,
            'n_estimators': n_est,
            'rmse': rmse_val
        })
        
        # Print progress for first and last n_estimators, and every 50th
        if n_est == 10 or n_est == 200 or n_est % 50 == 0:
            print(f"     n_estimators={n_est:3d} → RMSE: {rmse_val:.6f}")
    
    # Calculate mean RMSE for this max_depth
    mean_rmse = np.mean(rmse_values_for_depth)
    print(f"   ✓ max_depth={max_d} complete. Mean RMSE: {mean_rmse:.6f}")

print("\n   ✓ Training complete for all combinations!")

# Convert to DataFrame for easier analysis
results_df = pd.DataFrame(results)

# ==============================================================================
# STEP 3: CALCULATE MEAN RMSE FOR EACH max_depth
# ==============================================================================
print("\n🔍 Step 3: Calculating mean RMSE for each max_depth...")
print("   " + "-" * 80)

mean_rmse_by_depth = results_df.groupby('max_depth')['rmse'].mean().reset_index()
mean_rmse_by_depth.columns = ['max_depth', 'mean_rmse']
mean_rmse_by_depth = mean_rmse_by_depth.sort_values('mean_rmse')

print("\n   Mean RMSE for each max_depth:")
print("   " + "-" * 80)
print(f"   {'max_depth':>12} | {'Mean RMSE':>15}")
print("   " + "-" * 80)

for idx, row in mean_rmse_by_depth.iterrows():
    marker = " ⭐ BEST" if idx == mean_rmse_by_depth.index[0] else ""
    print(f"   {int(row['max_depth']):>12} | {row['mean_rmse']:>15.6f}{marker}")

best_max_depth = int(mean_rmse_by_depth.iloc[0]['max_depth'])
best_mean_rmse = mean_rmse_by_depth.iloc[0]['mean_rmse']

print(f"\n   🎯 Best max_depth: {best_max_depth} (Mean RMSE: {best_mean_rmse:.6f})")

# ==============================================================================
# STEP 4: DETAILED ANALYSIS
# ==============================================================================
print("\n" + "="*80)
print("📊 DETAILED RESULTS")
print("="*80)

print("\n   Summary statistics for each max_depth:")
print("   " + "-" * 80)

for max_d in max_depth_values:
    depth_data = results_df[results_df['max_depth'] == max_d]['rmse']
    mean_rmse = depth_data.mean()
    std_rmse = depth_data.std()
    min_rmse = depth_data.min()
    max_rmse = depth_data.max()
    
    marker = " ⭐" if max_d == best_max_depth else ""
    print(f"\n   max_depth={max_d}{marker}:")
    print(f"      Mean RMSE:  {mean_rmse:.6f}")
    print(f"      Std RMSE:   {std_rmse:.6f}")
    print(f"      Min RMSE:   {min_rmse:.6f}")
    print(f"      Max RMSE:   {max_rmse:.6f}")

# Show sample of RMSE values for each max_depth
print("\n   Sample RMSE values (showing every 4th n_estimators value):")
print("   " + "-" * 80)

for max_d in max_depth_values:
    depth_results = results_df[results_df['max_depth'] == max_d].sort_values('n_estimators')
    sample_results = depth_results[::4]  # Every 4th value
    
    marker = " ⭐" if max_d == best_max_depth else ""
    print(f"\n   max_depth={max_d}{marker}:")
    print(f"      {'n_estimators':>12} | {'RMSE':>15}")
    print("      " + "-" * 30)
    for idx, row in sample_results.iterrows():
        print(f"      {int(row['n_estimators']):>12} | {row['rmse']:>15.6f}")

# ==============================================================================
# STEP 5: COMPARE WITH ANSWER OPTIONS
# ==============================================================================
print("\n" + "="*80)
print("🎯 ANSWER ANALYSIS")
print("="*80)

options = [10, 15, 20, 25]

print(f"\n   Best max_depth (based on mean RMSE): {best_max_depth}")
print(f"\n   Multiple choice options:")
print("   " + "-" * 80)
print(f"   {'Option':>12} | {'Mean RMSE':>15} | {'Status':>12}")
print("   " + "-" * 80)

for opt in options:
    match = "✅ BEST" if opt == best_max_depth else "  "
    opt_data = mean_rmse_by_depth[mean_rmse_by_depth['max_depth'] == opt]
    if len(opt_data) > 0:
        mean_rmse_opt = opt_data.iloc[0]['mean_rmse']
        print(f"   {opt:>12} | {mean_rmse_opt:>15.6f} | {match:>12}")

print(f"\n   🎯 Answer: {best_max_depth}")

# ==============================================================================
# SUMMARY
# ==============================================================================
print("\n" + "="*80)
print("✨ SUMMARY")
print("="*80)

print(f"\n   Question: What's the best max_depth, using the mean RMSE?")
print(f"\n   ✅ Answer: {best_max_depth}")
print(f"\n   Explanation:")
print(f"      • Tested max_depth values: {max_depth_values}")
print(f"      • For each max_depth, tested n_estimators from 10 to 200 (step 10)")
print(f"      • Calculated mean RMSE across all n_estimators for each max_depth")
print(f"      • Best max_depth: {best_max_depth} with mean RMSE: {best_mean_rmse:.6f}")

# Show all mean RMSE values
print(f"\n   Mean RMSE by max_depth:")
for idx, row in mean_rmse_by_depth.iterrows():
    marker = " ⭐ BEST" if row['max_depth'] == best_max_depth else ""
    print(f"      max_depth={int(row['max_depth']):2d}: {row['mean_rmse']:.6f}{marker}")

print("\n" + "="*80)
print("✨ Analysis Complete!")
print("="*80)

