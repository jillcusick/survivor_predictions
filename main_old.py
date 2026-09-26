# Run pipeline to clean Survivor cast data, engineer features, and train and evaluate models

import pandas as pd
from src.config import (
    CLASSIFICATION_MODELS,
    REGRESSION_MODELS,
    SPLIT_SEASON,
    FILE_PATH
)
from src.ingest import load_and_clean_data
from src.engineer import engineer_features
from src.evaluate import evaluate_classification, evaluate_regression
from src.split import split_data
from src.train import train_model

def main():
    print("SURVIVOR PREDICTIONS PIPELINE")
    print("")

    # Load and clean raw dataset
    df = load_and_clean_data(FILE_PATH)

    # Engineer features and targets
    df_engineered = engineer_features(df)
    
    # Save processed dataset to CSV for reference
    df_engineered.to_csv("data/processed/survivor_engineered.csv", index=False)

    # Conduct train-test split
    X_train, X_test, y_train_class, y_test_class, y_train_reg, y_test_reg = (
        split_data(df_engineered, split_season=SPLIT_SEASON)
    )

    # Run regression models 
    
    print("REGRESSION EVALUATION")
    print("")
    
    for name, model in REGRESSION_MODELS.items():
        # Train
        pipeline = train_model(X_train, y_train_reg, model, step_name="regressor")
        # Evaluate
        evaluate_regression(pipeline, X_test, y_test_reg, model_name=name)

    # Run classification models
    
    print("CLASSIFICATION EVALUATION")
    print("")
    
    for name, model in CLASSIFICATION_MODELS.items():
        # Train
        pipeline = train_model(
            X_train, y_train_class, model, step_name="classifier"
        )
        # Evaluate
        evaluate_classification(pipeline, X_test, y_test_class, model_name=name)

    # Confirm end of pipeline
    print("PIPELINE EXECUTION COMPLETE")

if __name__ == "__main__":
    main()