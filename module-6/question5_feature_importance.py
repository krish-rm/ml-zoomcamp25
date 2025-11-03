"""
Question 5: Find the Most Important Feature

We can extract feature importance information from tree-based models.
At each step of the decision tree learning algorithm, it finds the best split.
When doing it, we can calculate "gain" - the reduction in impurity before and after the split.
This gain is quite useful in understanding what are the important features for tree-based models.

In Scikit-Learn, tree-based models contain this information in the feature_importances_ field.

For this homework question, we'll find the most important feature:

Train the model with these parameters:
- n_estimators=10
- max_depth=20
- random_state=1
- n_jobs=-1 (optional)

Get the feature importance information from this model.

What's the most important feature (among these 4)?
- vehicle_weight
- horsepower
- acceleration
- engine_displacement
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
print("QUESTION 5: FINDING THE MOST IMPORTANT FEATURE")
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
# STEP 2: TRAIN RANDOM FOREST MODEL
# ==============================================================================
print("\n🌲 Step 2: Training Random Forest Regressor...")
print("   Parameters:")
print("     • n_estimators=10")
print("     • max_depth=20")
print("     • random_state=1")
print("     • n_jobs=-1")

rf = RandomForestRegressor(
    n_estimators=10,
    max_depth=20,
    random_state=1,
    n_jobs=-1
)

rf.fit(X_train, y_train)
print("   ✓ Training complete!")

# Calculate RMSE for verification
y_val_pred = rf.predict(X_val)
rmse_val = np.sqrt(mean_squared_error(y_val, y_val_pred))
print(f"   ✓ Validation RMSE: {rmse_val:.6f}")

# ==============================================================================
# STEP 3: EXTRACT FEATURE IMPORTANCE
# ==============================================================================
print("\n📊 Step 3: Extracting feature importance...")
print("   " + "-" * 80)

feature_names = dv.get_feature_names_out()
feature_importance = rf.feature_importances_

# Create a DataFrame for easier analysis
importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': feature_importance
}).sort_values('importance', ascending=False)

print(f"\n   All features and their importance (sorted):")
print("   " + "-" * 80)
print(f"   {'Feature':<40} | {'Importance':>15}")
print("   " + "-" * 80)

for idx, row in importance_df.iterrows():
    print(f"   {row['feature']:<40} | {row['importance']:>15.6f}")

# ==============================================================================
# STEP 4: FIND THE MOST IMPORTANT FEATURE AMONG THE 4 SPECIFIED
# ==============================================================================
print("\n🔍 Step 4: Finding the most important feature among the 4 specified...")
print("   " + "-" * 80)

target_features = ['vehicle_weight', 'horsepower', 'acceleration', 'engine_displacement']

print(f"\n   Target features to check: {target_features}")

# Extract importance for each target feature
feature_importance_dict = {}
for feat in target_features:
    # Check if feature name appears in the vectorized feature names
    matching_features = [f for f in feature_names if feat in f]
    
    if matching_features:
        # If there are multiple matches (e.g., categorical features with values),
        # sum their importances. Otherwise, just get the importance.
        total_importance = sum(importance_df[importance_df['feature'].isin(matching_features)]['importance'].values)
        feature_importance_dict[feat] = total_importance
        print(f"   ✓ Found '{feat}' with importance: {total_importance:.6f}")
        if len(matching_features) > 1:
            print(f"     (Combined importance from {len(matching_features)} feature variants)")
    else:
        # Try exact match (for numeric features)
        exact_match = importance_df[importance_df['feature'] == feat]
        if len(exact_match) > 0:
            importance_value = exact_match.iloc[0]['importance']
            feature_importance_dict[feat] = importance_value
            print(f"   ✓ Found '{feat}' (exact match) with importance: {importance_value:.6f}")
        else:
            print(f"   ⚠ Warning: '{feat}' not found in feature names")
            feature_importance_dict[feat] = 0.0

# Find the most important feature
most_important_feature = max(feature_importance_dict, key=feature_importance_dict.get)
most_important_value = feature_importance_dict[most_important_feature]

print(f"\n   Importance comparison:")
print("   " + "-" * 80)
print(f"   {'Feature':<25} | {'Importance':>15} | {'Status':>12}")
print("   " + "-" * 80)

for feat in sorted(feature_importance_dict, key=feature_importance_dict.get, reverse=True):
    marker = " ✅ MOST IMPORTANT" if feat == most_important_feature else ""
    print(f"   {feat:<25} | {feature_importance_dict[feat]:>15.6f}{marker}")

# ==============================================================================
# STEP 5: ANSWER ANALYSIS
# ==============================================================================
print("\n" + "="*80)
print("🎯 ANSWER ANALYSIS")
print("="*80)

print(f"\n   Most important feature among the 4 specified: {most_important_feature}")
print(f"   Importance value: {most_important_value:.6f}")

print(f"\n   Multiple choice options:")
print("   " + "-" * 80)
for feat in target_features:
    marker = " ✅ ANSWER" if feat == most_important_feature else "  "
    print(f"   {marker} {feat:<25} | Importance: {feature_importance_dict[feat]:.6f}")

# ==============================================================================
# SUMMARY
# ==============================================================================
print("\n" + "="*80)
print("✨ SUMMARY")
print("="*80)

print(f"\n   Question: What's the most important feature (among these 4)?")
print(f"   Options: {', '.join(target_features)}")
print(f"\n   ✅ Answer: {most_important_feature}")
print(f"\n   Explanation:")
print(f"      • Trained Random Forest with n_estimators=10, max_depth=20, random_state=1")
print(f"      • Extracted feature importance from feature_importances_")
print(f"      • Compared importance values for the 4 specified features")
print(f"      • {most_important_feature} has the highest importance: {most_important_value:.6f}")

print("\n" + "="*80)
print("✨ Analysis Complete!")
print("="*80)

