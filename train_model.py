"""Train a pipeline model for garment productivity."""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from feature_engine.outliers import Winsorizer
import joblib

# Note: training research is on jupyter notebook and posted on medium
#  Medium post: https://medium.com/@kaikuh/machine-learning-code-focus-1bf13c848bc7


# Load dataset from GitHub
url = (
    "https://raw.githubusercontent.com/"
    "Alyxx-The-Sniper/Grament_production_analyis/"
    "refs/heads/main/garments_worker_productivity.csv"
)
df = pd.read_csv(url)

# Clean department names and filter sewing department
df["department"] = (
    df["department"]
    .str.strip()
    .replace({"sweing": "sewing", "finishing ": "finishing"})
)
df_sewing = df[df["department"] == "sewing"].reset_index(drop=True)

# Split features and target
X = df_sewing.drop(columns=["actual_productivity"])
y = df_sewing["actual_productivity"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Numeric preprocessing pipeline
numeric_pipeline = Pipeline(
    [
        ("winsor", Winsorizer(capping_method="iqr", tail="both", fold=1.5)),
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]
)

preprocessor = ColumnTransformer(
    [("num", numeric_pipeline, ["incentive"])]
)

# XGBoost model
model = XGBRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    verbosity=0,
)

# Full pipeline
pipeline = Pipeline(
    [("preprocessor", preprocessor), ("regressor", model)]
)

pipeline.fit(X_train, y_train)

# Save model to file
joblib.dump(pipeline, "model_1.pkl")
print("Model trained and saved as model_1.pkl")
