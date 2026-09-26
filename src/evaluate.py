import os
import json
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
    roc_curve,
)

def extract_and_plot_features(pipeline, X_test, model_name="Model", top_n=6):
    """Extracts coefficients or feature importance, prints them,
    and saves as chart in outputs folder.
    """
    model = pipeline[-1]
    feature_names = (
        X_test.columns
        if hasattr(X_test, "columns")
        else [f"feat_{i}" for i in range(X_test.shape[1])]
    )

    # Check what type of attribute the model supports
    if hasattr(model, "coef_"):
        coefs = model.coef_
        if coefs.ndim > 1:
            coefs = coefs[0]  # Flatten for binary classification
        ranked = sorted(
            zip(feature_names, coefs), key=lambda x: abs(x[1]), reverse=True
        )
        metric_label = "Coefficient Value"
    elif hasattr(model, "feature_importances_"):
        ranked = sorted(
            zip(feature_names, model.feature_importances_),
            key=lambda x: x[1],
            reverse=True,
        )
        metric_label = "Feature Importance"
    else:
        print(
            f"{model_name} does not support coefficient/importance lookup"
        )
        return

    # Print to terminal
    print(f"--- Feature Importance for {model_name} ---")
    for feat, val in ranked[:top_n]:
        print(f"  {feat}: {val:.4f}")
    print()

    # Slice top N for plotting
    top_features = ranked[:top_n]
    # Reverse order so highest magnitude appears at top of chart
    top_features.reverse()

    feats = [x[0] for x in top_features]
    vals = [x[1] for x in top_features]

    # Generate and save plot
    os.makedirs("outputs", exist_ok=True)
    safe_name = model_name.lower().replace(" ", "_")

    plt.figure(figsize=(8, max(5, top_n * 0.4)))
    bars = plt.barh(feats, vals, color="teal" if "Regression" in model_name or "Ridge" in model_name else "darkorange")
    
    plt.axvline(0, color="grey", linestyle="--", linewidth=0.8)
    plt.title(f"Top Drivers: {model_name}", fontsize=12, fontweight="bold")
    plt.xlabel(metric_label, fontsize=10)
    plt.ylabel("Features", fontsize=10)
    plt.tight_layout()
    
    plt.savefig(f"outputs/{safe_name}_feature_importance.png", dpi=300)
    plt.close()

def evaluate_regression(pipeline, X_test, y_test, model_name="Model"):
    """Generates regression model predictions on test data, evaluates model,
    outputs performance metrics, and saves visual artifacts (Actual vs. Predicted & Features).
    """
    preds = pipeline.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    rmse = mean_squared_error(y_test, preds) ** 0.5

    print(f"--- {model_name} (Regression) ---")
    print(f"MAE:  {mae:.4f}")
    print(f"RMSE: {rmse:.4f}\n")

    os.makedirs("outputs", exist_ok=True)
    safe_name = model_name.lower().replace(" ", "_")

    # Scatter plot of actual vs. predicted player placement score
    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, preds, alpha=0.6, color="teal", edgecolors="k")
    min_val = min(min(y_test), min(preds))
    max_val = max(max(y_test), max(preds))
    plt.plot([min_val, max_val], [min_val, max_val], "r--", lw=2, label="Perfect")
    plt.title(f"Actual vs Predicted: {model_name}")
    plt.xlabel("Actual Placement Score")
    plt.ylabel("Predicted Placement Score")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"outputs/{safe_name}_actual_vs_predicted.png")
    plt.close()

    # Print Ridge alpha determined with CV if applicable
    try:
        print(f"Selected Alpha for {model_name}: {pipeline[-1].alpha_}\n")
    except AttributeError:
        pass

    # Extract and save coefficient/feature importance plot
    extract_and_plot_features(pipeline, X_test, model_name, top_n=10)

    metrics = {"model_name": model_name, "mae": mae, "rmse": rmse}
    return preds, metrics

def evaluate_classification(pipeline, X_test, y_test, model_name="Model"):
    """Generates classification model predictions on test data, evaluates model,
    outputs performance metrics, and saves visual artifacts (Confusion Matrix, ROC, & Features).
    """
    preds = pipeline.predict(X_test)
    probs = (
        pipeline.predict_proba(X_test)[:, 1]
        if hasattr(pipeline, "predict_proba")
        else preds
    )

    acc = accuracy_score(y_test, preds)
    try:
        roc_auc = roc_auc_score(y_test, probs)
    except ValueError:
        roc_auc = 0.50

    print(f"--- {model_name} (Classification) ---")
    print(f"Accuracy: {acc:.4f}")
    print(f"ROC-AUC:  {roc_auc:.4f}\n")

    os.makedirs("outputs", exist_ok=True)
    safe_name = model_name.lower().replace(" ", "_")

    # Confusion matrix plot
    plt.figure(figsize=(6, 5))
    cm = confusion_matrix(y_test, preds)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title(f"Confusion Matrix: {model_name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(f"outputs/{safe_name}_confusion_matrix.png")
    plt.close()

    # ROC curve plot
    if hasattr(pipeline, "predict_proba"):
        fpr, tpr, _ = roc_curve(y_test, probs)
        plt.figure(figsize=(6, 5))
        plt.plot(
            fpr,
            tpr,
            color="darkorange",
            lw=2,
            label=f"ROC curve (AUC = {roc_auc:.2f})",
        )
        plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC Curve: {model_name}")
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(f"outputs/{safe_name}_roc_curve.png")
        plt.close()

    # Extract and save feature importance/coefficients plot
    extract_and_plot_features(pipeline, X_test, model_name, top_n=6)

    metrics = {"model_name": model_name, "accuracy": acc, "roc_auc": roc_auc}
    return preds, metrics


# Example of how you can collect and save all metrics into one file
def save_eval_metrics(all_metrics_list):
    os.makedirs("outputs/metrics", exist_ok=True)
    with open("outputs/metrics/evaluation_metrics.json", "w") as f:
        json.dump(all_metrics_list, f, indent=4)