import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
import joblib

# Load final data
df = pd.read_csv(r"C:\ML Projects\AI Project\AI-Powered-Flood-Risk-Assistant\data\preprocessed\kerala_flood_final_withRainfall_WITH_LABELS.csv")

# ✅ Define geographic context features based on domain knowledge
district_info = {
    "Thiruvananthapuram": {"coastal": True, "hilly": False, "riverine": True},
    "Kollam": {"coastal": True, "hilly": False, "riverine": True},
    "Pathanamthitta": {"coastal": False, "hilly": True, "riverine": True},
    "Alappuzha": {"coastal": True, "hilly": False, "riverine": True},
    "Kottayam": {"coastal": False, "hilly": False, "riverine": True},
    "Idukki": {"coastal": False, "hilly": True, "riverine": True},
    "Ernakulam": {"coastal": False, "hilly": False, "riverine": True},
    "Thrissur": {"coastal": False, "hilly": False, "riverine": True},
    "Palakkad": {"coastal": False, "hilly": True, "riverine": True},
    "Malappuram": {"coastal": False, "hilly": True, "riverine": True},
    "Kozhikode": {"coastal": True, "hilly": False, "riverine": True},
    "Wayanad": {"coastal": False, "hilly": True, "riverine": False},
    "Kannur": {"coastal": True, "hilly": False, "riverine": False},
    "Kasaragod": {"coastal": True, "hilly": False, "riverine": False}
}

# Add geographic features to dataframe
df['is_coastal'] = df['district'].map(lambda x: district_info.get(x, {}).get('coastal', False)).astype(int)
df['is_hilly'] = df['district'].map(lambda x: district_info.get(x, {}).get('hilly', False)).astype(int)
df['is_riverine'] = df['district'].map(lambda x: district_info.get(x, {}).get('riverine', False)).astype(int)

# FEATURES: ONLY PHYSICAL + GEOGRAPHIC — NO LATITUDE OR LONGITUDE
feature_cols = [
    'distance_to_river_km',
    'avg_annual_rainfall_mm',
    'population_density',
    'is_coastal',
    'is_hilly',
    'is_riverine'
]

# Prepare X, y
X = df[feature_cols].fillna(0)  # Fill any remaining NaN with 0
y = df['flooded_2018']          # REAL flood events from 2018 — this is our ground truth

# Log transform population density (helps with skew)
X['population_density_log'] = np.log1p(X['population_density'])
X = X.drop(['population_density'], axis=1)  # Drop raw version

# Final feature list after transformation — MUST MATCH APP EXACTLY
final_feature_names = [
    'distance_to_river_km',
    'avg_annual_rainfall_mm',
    'population_density_log',
    'is_coastal',
    'is_hilly',
    'is_riverine'
]
X.columns = final_feature_names 

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

# STRATIFIED K-FOLD CROSS-VALIDATION (5-fold) — TO AVOID OVERFITTING
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(
    RandomForestClassifier(
        n_estimators=100,           # Reduced to prevent overfitting
        max_depth=8,                # Shallower tree
        min_samples_split=10,       # Require at least 10 samples to split
        min_samples_leaf=5,         # Minimum 5 samples in leaf
        random_state=42,
        class_weight='balanced',
        oob_score=True              # Use out-of-bag score for internal validation
    ),
    X_scaled, y, cv=skf, scoring='accuracy'
)

print("Cross-Validation Results (5-Fold):")
print(f"Mean CV Accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
print(f"CV Scores per fold: {cv_scores}")

# Now train final model on full dataset (with best params)
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=8,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42,
    class_weight='balanced',
    oob_score=True
)

model.fit(X_scaled, y)

# Evaluate on training set 
y_pred_train = model.predict(X_scaled)
train_acc = accuracy_score(y, y_pred_train)
print(f"\nTraining Accuracy: {train_acc:.3f}")
print(f"Out-of-Bag Score: {model.oob_score_:.3f}")

# Split for final holdout test (smaller size to simulate real-world testing)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, stratify=y, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print(f"\nFinal Test Accuracy (on held-out 20%): {accuracy_score(y_test, y_pred):.3f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["High Risk", "Low Risk"]))

# Feature importance
importances = model.feature_importances_
print("\nTop Feature Importances (Ranked):")
for feat, imp in sorted(zip(final_feature_names, importances), key=lambda x: x[1], reverse=True):
    print(f"{feat}: {imp:.4f}")

# Save artifacts — these will be loaded by app.py
joblib.dump(model, 'flood_model_v2.pkl')
joblib.dump(scaler, 'scaler_v2.pkl')
joblib.dump(final_feature_names, 'feature_names.pkl')

print("\nArtifacts saved: flood_model_v2.pkl, scaler_v2.pkl, feature_names.pkl")
