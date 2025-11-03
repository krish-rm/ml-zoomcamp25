"""
Question 2: Train Random Forest Regressor and Calculate RMSE on Validation Set

Train a random forest regressor with:
- n_estimators=10
- random_state=1
- n_jobs=-1 (optional - to make training faster)

What's the RMSE of this model on the validation data?
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
print("QUESTION 2: RANDOM FOREST REGRESSOR")
print("="*80)

# Load and prepare data
print("\n📝 Step 1: Loading and preparing data...")
df = pd.read_csv('car_fuel_efficiency.csv')
df_prep = df.copy()
df_prep = df_prep.fillna(0)  # Fill missing values with zeros

# Separate target
y = df_prep['fuel_efficiency_mpg'].values
X_full = df_prep.drop('fuel_efficiency_mpg', axis=1)

# Train/validation/test split (60/20/20)
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X_full, y, test_size=0.2, random_state=1
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full, test_size=0.25, random_state=1  # 0.25 of 0.8 = 0.2
)

print(f"   ✓ Train set: {len(X_train)} samples (60%)")
print(f"   ✓ Validation set: {len(X_val)} samples (20%)")
print(f"   ✓ Test set: {len(X_test)} samples (20%)")

# Use DictVectorizer
print("\n📝 Step 2: Vectorizing features...")
dv = DictVectorizer(sparse=True)
train_dict = X_train.to_dict(orient='records')
X_train = dv.fit_transform(train_dict)

val_dict = X_val.to_dict(orient='records')
X_val = dv.transform(val_dict)

print(f"   ✓ Feature matrix shape: {X_train.shape}")

# Train Random Forest Regressor
print("\n🌲 Step 3: Training Random Forest Regressor...")
print("   Parameters:")
print("     • n_estimators=10")
print("     • random_state=1")
print("     • n_jobs=-1")

rf = RandomForestRegressor(
    n_estimators=10,
    random_state=1,
    n_jobs=-1
)

rf.fit(X_train, y_train)
print("   ✓ Training complete!")

# Make predictions
print("\n📊 Step 4: Making predictions...")
y_train_pred = rf.predict(X_train)
y_val_pred = rf.predict(X_val)

# Calculate RMSE
rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))
rmse_val = np.sqrt(mean_squared_error(y_val, y_val_pred))

print(f"\n✅ RESULTS:")
print("="*80)
print(f"   RMSE on Training Set:  {rmse_train:.6f}")
print(f"   RMSE on Validation Set: {rmse_val:.6f}")


# Additional analysis
print("\n" + "="*80)
print("📈 ADDITIONAL ANALYSIS:")
print("="*80)

print(f"\n   • Mean target value (validation): {np.mean(y_val):.4f} mpg")
print(f"   • Std target value (validation): {np.std(y_val):.4f} mpg")
print(f"   • RMSE as % of mean: {(rmse_val / np.mean(y_val)) * 100:.2f}%")
print(f"   • RMSE as % of std: {(rmse_val / np.std(y_val)) * 100:.2f}%")

# Feature importance
print(f"\n   • Top 5 most important features:")
feature_names = dv.get_feature_names_out()
feature_importance = rf.feature_importances_
important_features = sorted(zip(feature_names, feature_importance), 
                           key=lambda x: x[1], reverse=True)
for i, (feat, imp) in enumerate(important_features[:5], 1):
    print(f"     {i}. {feat:30s} | Importance: {imp:.4f}")

print("\n" + "="*80)
print("✨ Analysis Complete!")
print("="*80)

