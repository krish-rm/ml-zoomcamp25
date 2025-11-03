"""
Question 6: Tune the eta parameter in XGBoost

Now let's train an XGBoost model! For this question, we'll tune the eta parameter:

- Install XGBoost
- Create DMatrix for train and validation
- Create a watchlist
- Train a model with these parameters for 100 rounds:

xgb_params = {
    'eta': 0.3, 
    'max_depth': 6,
    'min_child_weight': 1,
    
    'objective': 'reg:squarederror',
    'nthread': 8,
    
    'seed': 1,
    'verbosity': 1,
}

Now change eta from 0.3 to 0.1.

Which eta leads to the best RMSE score on the validation dataset?
- 0.3
- 0.1
- Both give equal value
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# Try to import xgboost, install if not available
try:
    import xgboost as xgb
    print("✓ XGBoost is already installed")
except ImportError:
    print("📦 Installing XGBoost...")
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "xgboost"])
    import xgboost as xgb
    print("✓ XGBoost installed successfully")

print("="*80)
print("QUESTION 6: TUNING ETA PARAMETER IN XGBOOST")
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
X_train_sparse = dv.fit_transform(train_dict)

val_dict = X_val.to_dict(orient='records')
X_val_sparse = dv.transform(val_dict)

print(f"   ✓ Feature matrix shape: {X_train_sparse.shape}")

# ==============================================================================
# STEP 2: CREATE DMATRIX FOR TRAIN AND VALIDATION
# ==============================================================================
print("\n📊 Step 2: Creating DMatrix for train and validation...")

# Convert sparse matrices to dense for XGBoost (or XGBoost can handle sparse)
# XGBoost can work with scipy sparse matrices directly
dtrain = xgb.DMatrix(X_train_sparse, label=y_train)
dval = xgb.DMatrix(X_val_sparse, label=y_val)

print("   ✓ DMatrix created for training set")
print("   ✓ DMatrix created for validation set")

# ==============================================================================
# STEP 3: CREATE WATCHLIST
# ==============================================================================
print("\n👀 Step 3: Creating watchlist...")

watchlist = [(dtrain, 'train'), (dval, 'val')]
print(f"   ✓ Watchlist created with {len(watchlist)} datasets")

# ==============================================================================
# STEP 4: TRAIN MODEL WITH eta=0.3
# ==============================================================================
print("\n🚀 Step 4: Training XGBoost model with eta=0.3...")
print("   " + "-" * 80)

xgb_params_03 = {
    'eta': 0.3, 
    'max_depth': 6,
    'min_child_weight': 1,
    
    'objective': 'reg:squarederror',
    'nthread': 8,
    
    'seed': 1,
    'verbosity': 1,
}

print("\n   Parameters:")
for key, value in xgb_params_03.items():
    print(f"     • {key}: {value}")

print("\n   Training for 100 rounds...")
model_03 = xgb.train(
    xgb_params_03,
    dtrain,
    num_boost_round=100,
    evals=watchlist,
    verbose_eval=False
)

# Make predictions and calculate RMSE
y_val_pred_03 = model_03.predict(dval)
rmse_val_03 = np.sqrt(mean_squared_error(y_val, y_val_pred_03))

print(f"\n   ✓ Training complete!")
print(f"   ✓ Validation RMSE (eta=0.3): {rmse_val_03:.6f}")

# ==============================================================================
# STEP 5: TRAIN MODEL WITH eta=0.1
# ==============================================================================
print("\n🚀 Step 5: Training XGBoost model with eta=0.1...")
print("   " + "-" * 80)

xgb_params_01 = {
    'eta': 0.1, 
    'max_depth': 6,
    'min_child_weight': 1,
    
    'objective': 'reg:squarederror',
    'nthread': 8,
    
    'seed': 1,
    'verbosity': 1,
}

print("\n   Parameters:")
for key, value in xgb_params_01.items():
    print(f"     • {key}: {value}")

print("\n   Training for 100 rounds...")
model_01 = xgb.train(
    xgb_params_01,
    dtrain,
    num_boost_round=100,
    evals=watchlist,
    verbose_eval=False
)

# Make predictions and calculate RMSE
y_val_pred_01 = model_01.predict(dval)
rmse_val_01 = np.sqrt(mean_squared_error(y_val, y_val_pred_01))

print(f"\n   ✓ Training complete!")
print(f"   ✓ Validation RMSE (eta=0.1): {rmse_val_01:.6f}")

# ==============================================================================
# STEP 6: COMPARE RESULTS
# ==============================================================================
print("\n" + "="*80)
print("📊 COMPARISON RESULTS")
print("="*80)

print(f"\n   Validation RMSE comparison:")
print("   " + "-" * 80)
print(f"   {'eta':>10} | {'Validation RMSE':>20} | {'Status':>20}")
print("   " + "-" * 80)

if rmse_val_03 < rmse_val_01:
    best_eta = 0.3
    best_rmse = rmse_val_03
    marker_03 = " ✅ BEST"
    marker_01 = ""
elif rmse_val_01 < rmse_val_03:
    best_eta = 0.1
    best_rmse = rmse_val_01
    marker_03 = ""
    marker_01 = " ✅ BEST"
else:
    best_eta = "Both (equal)"
    best_rmse = rmse_val_03
    marker_03 = " ✅ (Equal)"
    marker_01 = " ✅ (Equal)"

print(f"   {0.3:>10.1f} | {rmse_val_03:>20.6f}{marker_03}")
print(f"   {0.1:>10.1f} | {rmse_val_01:>20.6f}{marker_01}")

difference = abs(rmse_val_03 - rmse_val_01)
print(f"\n   Difference: {difference:.6f}")

# ==============================================================================
# STEP 7: DETAILED ANALYSIS
# ==============================================================================
print("\n" + "="*80)
print("🔍 DETAILED ANALYSIS")
print("="*80)

print(f"\n   Training details:")
print("   " + "-" * 80)
print(f"   • Both models trained for 100 rounds")
print(f"   • Both models use max_depth=6, min_child_weight=1")
print(f"   • Only difference: eta parameter")
print(f"   • Random seed: 1 (for reproducibility)")

print(f"\n   RMSE Analysis:")
print("   " + "-" * 80)
if rmse_val_03 < rmse_val_01:
    improvement = ((rmse_val_01 - rmse_val_03) / rmse_val_01) * 100
    print(f"   • eta=0.3 performs better by {improvement:.2f}%")
    print(f"   • RMSE improvement: {rmse_val_01 - rmse_val_03:.6f}")
elif rmse_val_01 < rmse_val_03:
    improvement = ((rmse_val_03 - rmse_val_01) / rmse_val_03) * 100
    print(f"   • eta=0.1 performs better by {improvement:.2f}%")
    print(f"   • RMSE improvement: {rmse_val_03 - rmse_val_01:.6f}")
else:
    print(f"   • Both eta values give exactly the same RMSE")

# ==============================================================================
# STEP 8: ANSWER ANALYSIS
# ==============================================================================
print("\n" + "="*80)
print("🎯 ANSWER ANALYSIS")
print("="*80)

print(f"\n   Best eta value (based on validation RMSE): {best_eta}")
print(f"\n   Multiple choice options:")
print("   " + "-" * 80)

options = [0.3, 0.1, "Both give equal value"]
for opt in options:
    if isinstance(opt, float):
        match = " ✅ ANSWER" if opt == best_eta or (best_eta == "Both (equal)" and opt in [0.3, 0.1]) else "  "
        rmse_value = rmse_val_03 if opt == 0.3 else rmse_val_01
        print(f"   {match} {opt} | RMSE: {rmse_value:.6f}")
    else:
        match = " ✅ ANSWER" if best_eta == "Both (equal)" else "  "
        print(f"   {match} {opt}")

# ==============================================================================
# SUMMARY
# ==============================================================================
print("\n" + "="*80)
print("✨ SUMMARY")
print("="*80)

print(f"\n   Question: Which eta leads to the best RMSE score on the validation dataset?")
print(f"\n   ✅ Answer: {best_eta}")
print(f"\n   Explanation:")
print(f"      • Trained XGBoost models with eta=0.3 and eta=0.1 for 100 rounds")
print(f"      • Used DMatrix for train and validation sets")
print(f"      • Created watchlist for monitoring training")
print(f"      • Calculated RMSE on validation set for both models")
if rmse_val_03 < rmse_val_01:
    print(f"      • eta=0.3 achieves better RMSE: {rmse_val_03:.6f} vs {rmse_val_01:.6f}")
elif rmse_val_01 < rmse_val_03:
    print(f"      • eta=0.1 achieves better RMSE: {rmse_val_01:.6f} vs {rmse_val_03:.6f}")
else:
    print(f"      • Both eta values achieve the same RMSE: {best_rmse:.6f}")

print("\n" + "="*80)
print("✨ Analysis Complete!")
print("="*80)

