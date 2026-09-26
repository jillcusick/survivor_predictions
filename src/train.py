from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_preprocessor():
    """Defines the ColumnTransformer for numerical and categorical features."""
    
    numerical_cols = ["rel_age"]
    categorical_cols = ["region", "industry", "gender", "new_era", "bipoc"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_cols),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical_cols,
            ),
        ],
        remainder="drop",
    )
    return preprocessor


def train_model(X_train, y_train, model, step_name="regressor"):
    """Builds the pipeline, fits it to the training data, and returns the trained pipeline."""
    
    # applies preprocessing to features 
    preprocessor = build_preprocessor()

    # builds the pipeline with preprocessing and the model
    pipeline = Pipeline(
        steps=[("preprocessor", preprocessor), (step_name, model)]
    )

    # fits model to the training data 
    pipeline.fit(X_train, y_train)
    return pipeline