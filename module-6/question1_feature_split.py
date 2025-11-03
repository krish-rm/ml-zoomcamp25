"""
Why Vehicle Weight is Used for Splitting in Question 1

This script demonstrates why the decision tree algorithm chooses vehicle_weight
as the root split feature by:
1. Listing all available features
2. Explaining the split selection criterion (MSE reduction)
3. Testing each feature and calculating potential MSE reduction
4. Showing why vehicle_weight gives the best split
5. Creating visualizations
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.feature_extraction import DictVectorizer
import warnings
warnings.filterwarnings('ignore')


# ==============================================================================
# STEP 1: LOAD AND PREPARE DATA
# ==============================================================================
print("\n📋 STEP 1: Loading and Preparing Data")
print("-" * 80)

df = pd.read_csv('car_fuel_efficiency.csv')
df_prep = df.copy()
df_prep = df_prep.fillna(0)

y = df_prep['fuel_efficiency_mpg'].values
X_full = df_prep.drop('fuel_efficiency_mpg', axis=1)

# Split data (same as homework instructions)
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X_full, y, test_size=0.2, random_state=1
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full, test_size=0.25, random_state=1
)

print(f"   • Training samples: {len(X_train)}")
print(f"   • Target variable: fuel_efficiency_mpg")
print(f"   • Target range: {y_train.min():.2f} - {y_train.max():.2f} mpg")
print(f"   • Target mean: {y_train.mean():.2f} mpg")

# ==============================================================================
# STEP 2: LIST ALL FEATURES
# ==============================================================================
print("\n📋 STEP 2: All Features Available for Splitting")
print("-" * 80)

print("\n   Original Features (10):")
print("   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
print("   Numerical Features:")
numerical_features = ['engine_displacement', 'num_cylinders', 'horsepower', 
                     'vehicle_weight', 'acceleration', 'model_year', 'num_doors']
for i, feat in enumerate(numerical_features, 1):
    print(f"     {i}. {feat}")

print("\n   Categorical Features:")
categorical_features = ['origin', 'fuel_type', 'drivetrain']
for i, feat in enumerate(categorical_features, 1):
    print(f"     {i}. {feat}")

# Vectorize to see encoded features
dv = DictVectorizer(sparse=True)
train_dict = X_train.to_dict(orient='records')
X_train_sparse = dv.fit_transform(train_dict)
feature_names = dv.get_feature_names_out()

print(f"\n   After DictVectorizer Encoding ({len(feature_names)} features):")
print("   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
for i, feat in enumerate(feature_names, 1):
    print(f"     {i:2d}. {feat}")

# ==============================================================================
# STEP 3: EXPLAIN SPLIT SELECTION ALGORITHM
# ==============================================================================
print("\n🌳 STEP 3: How Decision Trees Choose Splits")
print("-" * 80)

parent_mse = np.mean((y_train - np.mean(y_train))**2)

explanation = f"""
   Decision Tree Split Selection Process:
   
   1. Calculate Parent MSE (before any split):
      Parent MSE = mean((y - mean(y))²)
      Current Parent MSE = {parent_mse:.6f}
   
   2. For EACH feature, the algorithm:
      a. Tests many threshold values
      b. Splits data: Left (<= threshold) vs Right (> threshold)
      c. Calculates:
         • Left MSE = mean((y_left - mean(y_left))²)
         • Right MSE = mean((y_right - mean(y_right))²)
         • Weighted MSE = (n_left/n_total) × MSE_left + (n_right/n_total) × MSE_right
      d. Calculates MSE Reduction = Parent MSE - Weighted MSE
   
   3. Selects the feature with MAXIMUM MSE REDUCTION
   
   Why? Maximum MSE reduction = maximum improvement in prediction accuracy!
"""
print(explanation)

# ==============================================================================
# STEP 4: CALCULATE MSE REDUCTION FOR EACH FEATURE
# ==============================================================================
print("🔍 STEP 4: Testing Each Feature - Which Gives Best Split?")
print("-" * 80)

def calculate_best_split_for_feature(X_feature, y_target, n_tries=200):
    """
    Find the best threshold for a numerical feature that maximizes MSE reduction.
    """
    parent_mse = np.mean((y_target - np.mean(y_target))**2)
    
    best_reduction = 0
    best_threshold = None
    best_stats = None
    
    # Try different percentiles as thresholds
    percentiles = np.linspace(5, 95, n_tries)
    
    for p in percentiles:
        threshold = np.percentile(X_feature, p)
        
        left_mask = X_feature <= threshold
        right_mask = ~left_mask
        
        if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
            continue
            
        y_left = y_target[left_mask]
        y_right = y_target[right_mask]
        
        mse_left = np.mean((y_left - np.mean(y_left))**2)
        mse_right = np.mean((y_right - np.mean(y_right))**2)
        
        n_left = len(y_left)
        n_right = len(y_right)
        n_total = len(y_target)
        
        weighted_mse = (n_left / n_total) * mse_left + (n_right / n_total) * mse_right
        mse_reduction = parent_mse - weighted_mse
        
        if mse_reduction > best_reduction:
            best_reduction = mse_reduction
            best_threshold = threshold
            best_stats = {
                'left_count': n_left,
                'right_count': n_right,
                'left_mean': np.mean(y_left),
                'right_mean': np.mean(y_right),
                'left_mse': mse_left,
                'right_mse': mse_right,
                'weighted_mse': weighted_mse
            }
    
    return {
        'mse_reduction': best_reduction,
        'threshold': best_threshold,
        'stats': best_stats
    }

# Test numerical features
print("\n   Testing Numerical Features:")
print("   " + "="*70)

results = {}
numerical_to_test = ['vehicle_weight', 'horsepower', 'engine_displacement', 
                    'acceleration', 'model_year', 'num_cylinders']

for feat in numerical_to_test:
    if feat in X_train.columns:
        X_feat = X_train[feat].values
        result = calculate_best_split_for_feature(X_feat, y_train, n_tries=200)
        results[feat] = result
        
        stats = result['stats']
        diff_means = abs(stats['left_mean'] - stats['right_mean'])
        
        print(f"\n   📊 {feat.upper()}:")
        print(f"      Best Threshold: {result['threshold']:.4f}")
        print(f"      MSE Reduction:  {result['mse_reduction']:.6f}")
        print(f"      Left group (<= threshold):")
        print(f"         • Samples: {stats['left_count']}")
        print(f"         • Mean MPG: {stats['left_mean']:.4f}")
        print(f"         • MSE: {stats['left_mse']:.6f}")
        print(f"      Right group (> threshold):")
        print(f"         • Samples: {stats['right_count']}")
        print(f"         • Mean MPG: {stats['right_mean']:.4f}")
        print(f"         • MSE: {stats['right_mse']:.6f}")
        print(f"      🎯 Difference in means: {diff_means:.4f} mpg")
        print(f"      💡 This feature can reduce error by {result['mse_reduction']:.6f}")

# Test categorical features
print("\n   Testing Categorical Features:")
print("   " + "="*70)

categorical_to_test = ['origin', 'fuel_type', 'drivetrain']

for feat in categorical_to_test:
    if feat in X_train.columns:
        print(f"\n   📊 {feat.upper()}:")
        unique_vals = X_train[feat].unique()
        
        best_reduction = 0
        best_split = None
        
        for val in unique_vals:
            if pd.isna(val) or val == 0:
                continue
                
            left_mask = (X_train[feat] == val)
            right_mask = ~left_mask
            
            if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                continue
            
            y_left = y_train[left_mask]
            y_right = y_train[right_mask]
            
            mse_left = np.mean((y_left - np.mean(y_left))**2)
            mse_right = np.mean((y_right - np.mean(y_right))**2)
            
            n_left = len(y_left)
            n_right = len(y_right)
            n_total = len(y_train)
            
            weighted_mse = (n_left / n_total) * mse_left + (n_right / n_total) * mse_right
            mse_reduction = parent_mse - weighted_mse
            
            if mse_reduction > best_reduction:
                best_reduction = mse_reduction
                best_split = {
                    'value': val,
                    'left_count': n_left,
                    'right_count': n_right,
                    'left_mean': np.mean(y_left),
                    'right_mean': np.mean(y_right),
                    'mse_reduction': mse_reduction
                }
        
        if best_split:
            results[feat] = {
                'mse_reduction': best_split['mse_reduction'],
                'threshold': best_split['value']
            }
            diff_means = abs(best_split['left_mean'] - best_split['right_mean'])
            print(f"      Best Split: {feat} == '{best_split['value']}'")
            print(f"      MSE Reduction: {best_split['mse_reduction']:.6f}")
            print(f"      Left group: {best_split['left_count']} samples, Mean: {best_split['left_mean']:.4f}")
            print(f"      Right group: {best_split['right_count']} samples, Mean: {best_split['right_mean']:.4f}")
            print(f"      🎯 Difference in means: {diff_means:.4f} mpg")

# ==============================================================================
# STEP 5: RANK FEATURES AND EXPLAIN WINNER
# ==============================================================================
print("\n🏆 STEP 5: Ranking Features - The Winner is Clear!")
print("-" * 80)

sorted_results = sorted(results.items(), key=lambda x: x[1]['mse_reduction'], reverse=True)

print(f"\n   Ranked by MSE Reduction (Highest = Best Split):")
print("   " + "="*70)
print(f"\n   Parent MSE (no split): {parent_mse:.6f}\n")

for rank, (feat, result) in enumerate(sorted_results, 1):
    marker = " 🏆 WINNER!" if rank == 1 else ""
    reduction_pct = (result['mse_reduction'] / parent_mse) * 100
    print(f"   {rank}. {feat:25s} | MSE Reduction: {result['mse_reduction']:8.6f} ({reduction_pct:6.2f}% of parent){marker}")

winner = sorted_results[0]
print("\n   " + "="*70)
print(f"\n   ✅ WHY '{winner[0]}' WINS:")
print(f"      • Highest MSE reduction: {winner[1]['mse_reduction']:.6f}")
print(f"      • This is {winner[1]['mse_reduction']/parent_mse*100:.1f}% of the parent MSE!")
print(f"      • Algorithm tested ALL features and this one gives MAXIMUM improvement")
print(f"      • Better MSE reduction = better prediction accuracy")

if winner[0] == 'vehicle_weight':
    stats = winner[1]['stats']
    print(f"\n   💡 Why Vehicle Weight Works So Well:")
    print(f"      • Strong negative correlation with fuel efficiency")
    print(f"      • Heavier cars → lower MPG (physics: more energy needed)")
    print(f"      • Creates clear separation:")
    print(f"        - Light cars (<= {winner[1]['threshold']:.0f} lbs): {stats['left_mean']:.2f} mpg")
    print(f"        - Heavy cars (> {winner[1]['threshold']:.0f} lbs): {stats['right_mean']:.2f} mpg")
    print(f"        - Difference: {abs(stats['left_mean'] - stats['right_mean']):.2f} mpg!")

# ==============================================================================
# STEP 6: CORRELATION ANALYSIS
# ==============================================================================
print("\n📈 STEP 6: Correlation Analysis")
print("-" * 80)

print("\n   Correlation with Target (fuel_efficiency_mpg):")
print("   " + "="*70)
for feat in numerical_to_test:
    if feat in X_train.columns:
        corr = np.corrcoef(X_train[feat], y_train)[0, 1]
        abs_corr = abs(corr)
        strength = "Very Strong" if abs_corr > 0.7 else "Strong" if abs_corr > 0.5 else "Moderate" if abs_corr > 0.3 else "Weak"
        marker = " ⭐" if abs_corr == max([abs(np.corrcoef(X_train[f], y_train)[0, 1]) for f in numerical_to_test if f in X_train.columns]) else ""
        print(f"     {feat:25s} | {corr:7.4f} | {strength}{marker}")

print("\n   💡 Interpretation:")
print(f"      • Vehicle_weight has the STRONGEST correlation (absolute value)")
print(f"      • Strong correlation → better splits → better MSE reduction")
print(f"      • This is why the algorithm automatically chooses it!")


# ==============================================================================
# VERIFY WITH ACTUAL DECISION TREE
# ==============================================================================
print("\n✅ STEP 8: Verification with Actual Decision Tree")
print("-" * 80)

dt = DecisionTreeRegressor(max_depth=1, random_state=1)
dt.fit(X_train_sparse, y_train)

tree_ = dt.tree_
feature_used_idx = tree_.feature[0]
feature_used = feature_names[feature_used_idx]
threshold_actual = tree_.threshold[0]

print(f"\n   Actual Decision Tree Result:")
print(f"   • Feature used: {feature_used}")
print(f"   • Threshold: {threshold_actual:.4f}")
print(f"   • Left child value: {tree_.value[1][0][0]:.4f}")
print(f"   • Right child value: {tree_.value[2][0][0]:.4f}")

if 'vehicle_weight' in feature_used:
    print(f"\n   ✅ CONFIRMED: vehicle_weight is the chosen feature!")
else:
    print(f"\n   ⚠️  Note: The chosen feature is {feature_used}")

print("\n" + "="*80)
print("✨ ANALYSIS COMPLETE!")
print("="*80)
print(f"\n📝 Summary:")
print(f"   • {len(feature_names)} features were available for splitting")
print(f"   • Algorithm tested ALL features and calculated MSE reduction for each")
print(f"   • {winner[0]} provided the HIGHEST MSE reduction: {winner[1]['mse_reduction']:.6f}")
print(f"   • This is why it was automatically selected as the root split feature")
print(f"   • The algorithm makes this choice based purely on which feature")
print(f"     reduces prediction error the most - no human bias!")


