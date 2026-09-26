from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression, RidgeCV, LogisticRegressionCV 
from xgboost import XGBClassifier, XGBRegressor
from sklearn.dummy import DummyClassifier, DummyRegressor

# Global constants and random seeds
RANDOM_STATE = 42
SPLIT_SEASON = 45

# Data file path
FILE_PATH = "data/raw/survivoR.xlsx"

# Regression models (predicting relative placement scores)
REGRESSION_MODELS = {
    "Null Model (Mean Baseline)": DummyRegressor(strategy="mean"),
    "Linear Regression": LinearRegression(),
    # Tunes L2 regularization strength with CV 
    "Ridge Regression": RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]),
    # Tuned hyperparameters for prediction: restricted depth and leaf size
    "Random Forest Regressor": RandomForestRegressor(
        n_estimators=200,
        max_depth=10,
        min_samples_leaf=4,
        random_state=RANDOM_STATE,
    ),
    # Tuned hyperparameters for prediction: lower learning rate with more trees and subsampling
    "XGBoost Regressor": XGBRegressor(
        n_estimators=200,
        learning_rate=0.03,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_STATE,
    ),
}

# Classification Models (predicting whether players will make merge/jury)
CLASSIFICATION_MODELS = {
    "Null Model (Majority Class Baseline)": DummyClassifier(strategy="most_frequent"),
    # Tunes L2 regularization strength with CV 
    "Logistic Regression": LogisticRegression(
            C=1.0,  # standard regularization strength
            max_iter=1000,
            random_state=RANDOM_STATE,
            class_weight='balanced'
        ),
    # Tuned hyperparameters for prediction: restricted depth and leaf size
    "Random Forest Classifier": RandomForestClassifier(
        class_weight="balanced",
        n_estimators=200,
        max_depth=10,
        min_samples_leaf=4,
        random_state=RANDOM_STATE,
    ),
    # Tuned hyperparameters for prediction: lower learning rate with more trees and subsampling
    "XGBoost Classifier": XGBClassifier(
        n_estimators=200,
        learning_rate=0.03,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_STATE,
    ),
}