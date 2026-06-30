import logging
import os
import re
import warnings
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeRegressor


warnings.filterwarnings("ignore")

DATA_PATH = "data/clean_service_center_data.csv"
MODEL_DIR = "models"
LOG_DIR = "logs"

TICKET_MODEL_PATH = os.path.join(MODEL_DIR, "ticket_classifier.pkl")
REPAIR_MODEL_PATH = os.path.join(MODEL_DIR, "repair_prediction.pkl")
TFIDF_PATH = os.path.join(MODEL_DIR, "tfidf.pkl")
METRICS_PATH = os.path.join(MODEL_DIR, "model_metrics.csv")


def setup_logging() -> None:
    """Create logging configuration."""
    os.makedirs(LOG_DIR, exist_ok=True)

    logging.basicConfig(
        filename=os.path.join(LOG_DIR, "training.log"),
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


def create_directories() -> None:
    """Create required folders."""
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)


def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    """Load cleaned service center dataset."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset not found at {path}. Please run data.py first."
        )

    df = pd.read_csv(path)
    logging.info("Dataset loaded successfully. Shape: %s", df.shape)

    return df


def clean_text(text: str) -> str:
    """
    Clean complaint text.

    Steps:
    - Convert to lowercase
    - Remove special characters
    - Remove extra spaces
    """
    text = str(text).lower()
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def get_one_hot_encoder() -> OneHotEncoder:
    """
    Create OneHotEncoder.

    This supports both old and new scikit-learn versions.
    """
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def train_ticket_classifier(df: pd.DataFrame) -> Tuple[Pipeline, Dict[str, float]]:
    """
    Train complaint ticket classification model.

    Input  : Complaint text
    Output : Complaint_Type
    """

    print("\nTraining Ticket Classification Models...")
    logging.info("Started ticket classification training.")

    df = df.dropna(subset=["Complaint", "Complaint_Type"])

    x = df["Complaint"]
    y = df["Complaint_Type"]

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Linear SVM": LinearSVC(),
        "Naive Bayes": MultinomialNB(),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            random_state=42,
        ),
    }

    best_model = None
    best_model_name = ""
    best_f1 = 0.0
    results = []

    for model_name, model in models.items():
        pipeline = Pipeline(
            steps=[
                (
                    "tfidf",
                    TfidfVectorizer(
                        preprocessor=clean_text,
                        stop_words="english",
                        max_features=5000,
                        ngram_range=(1, 2),
                    ),
                ),
                ("model", model),
            ]
        )

        pipeline.fit(x_train, y_train)
        y_pred = pipeline.predict(x_test)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(
            y_test, y_pred, average="weighted", zero_division=0
        )
        recall = recall_score(
            y_test, y_pred, average="weighted", zero_division=0
        )
        f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

        results.append(
            {
                "Module": "Ticket Classification",
                "Model": model_name,
                "Accuracy": accuracy,
                "Precision": precision,
                "Recall": recall,
                "F1_Score": f1,
                "MAE": np.nan,
                "RMSE": np.nan,
                "R2_Score": np.nan,
            }
        )

        print(f"\n{model_name}")
        print(f"Accuracy : {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall   : {recall:.4f}")
        print(f"F1 Score : {f1:.4f}")

        logging.info(
            "%s - Accuracy: %.4f, F1: %.4f",
            model_name,
            accuracy,
            f1,
        )

        if f1 > best_f1:
            best_f1 = f1
            best_model = pipeline
            best_model_name = model_name

    if best_model is None:
        raise RuntimeError("Ticket classifier training failed.")

    y_best_pred = best_model.predict(x_test)

    report = classification_report(y_test, y_best_pred)
    matrix = confusion_matrix(y_test, y_best_pred)

    with open(os.path.join(MODEL_DIR, "classification_report.txt"), "w") as file:
        file.write(f"Best Model: {best_model_name}\n\n")
        file.write(report)
        file.write("\n\nConfusion Matrix:\n")
        file.write(str(matrix))

    joblib.dump(best_model, TICKET_MODEL_PATH)
    joblib.dump(best_model.named_steps["tfidf"], TFIDF_PATH)

    print("\nBest Ticket Classifier:", best_model_name)
    print("Saved:", TICKET_MODEL_PATH)
    print("Saved:", TFIDF_PATH)

    logging.info("Best ticket classifier saved: %s", best_model_name)

    best_result = {
        "best_ticket_model": best_model_name,
        "best_ticket_f1": best_f1,
        "results": results,
    }

    return best_model, best_result


def train_repair_time_model(df: pd.DataFrame) -> Tuple[Pipeline, Dict[str, float]]:
    """
    Train repair time prediction model.

    Target:
    - Repair_Time

    Features:
    - Complaint_Type
    - Brand
    - Priority
    - Warranty_Status
    - Parts_Replaced
    - Repair_Cost
    """

    print("\nTraining Repair Time Prediction Models...")
    logging.info("Started repair time prediction training.")

    required_columns = [
        "Complaint_Type",
        "Brand",
        "Priority",
        "Warranty_Status",
        "Parts_Replaced",
        "Repair_Cost",
        "Repair_Time",
    ]

    df = df.dropna(subset=required_columns)

    features = [
        "Complaint_Type",
        "Brand",
        "Priority",
        "Warranty_Status",
        "Parts_Replaced",
        "Repair_Cost",
    ]

    target = "Repair_Time"

    x = df[features]
    y = df[target]

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42,
    )

    categorical_features = [
        "Complaint_Type",
        "Brand",
        "Priority",
        "Warranty_Status",
        "Parts_Replaced",
    ]

    numerical_features = ["Repair_Cost"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", get_one_hot_encoder(), categorical_features),
            ("num", StandardScaler(), numerical_features),
        ]
    )

    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(random_state=42),
        "Random Forest": RandomForestRegressor(
            n_estimators=200,
            random_state=42,
        ),
        "Gradient Boosting": GradientBoostingRegressor(random_state=42),
    }

    best_model = None
    best_model_name = ""
    best_r2 = -999
    results = []

    for model_name, model in models.items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", model),
            ]
        )

        pipeline.fit(x_train, y_train)
        y_pred = pipeline.predict(x_test)

        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)

        results.append(
            {
                "Module": "Repair Time Prediction",
                "Model": model_name,
                "Accuracy": np.nan,
                "Precision": np.nan,
                "Recall": np.nan,
                "F1_Score": np.nan,
                "MAE": mae,
                "RMSE": rmse,
                "R2_Score": r2,
            }
        )

        print(f"\n{model_name}")
        print(f"MAE : {mae:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"R2  : {r2:.4f}")

        logging.info(
            "%s - MAE: %.4f, RMSE: %.4f, R2: %.4f",
            model_name,
            mae,
            rmse,
            r2,
        )

        if r2 > best_r2:
            best_r2 = r2
            best_model = pipeline
            best_model_name = model_name

    if best_model is None:
        raise RuntimeError("Repair time model training failed.")

    joblib.dump(best_model, REPAIR_MODEL_PATH)

    print("\nBest Repair Time Model:", best_model_name)
    print("Saved:", REPAIR_MODEL_PATH)

    logging.info("Best repair time model saved: %s", best_model_name)

    best_result = {
        "best_repair_model": best_model_name,
        "best_repair_r2": best_r2,
        "results": results,
    }

    return best_model, best_result


def save_metrics(
    ticket_result: Dict[str, float],
    repair_result: Dict[str, float],
) -> None:
    """Save model evaluation results into CSV."""

    all_results = ticket_result["results"] + repair_result["results"]
    metrics_df = pd.DataFrame(all_results)
    metrics_df.to_csv(METRICS_PATH, index=False)

    print("\nModel metrics saved:", METRICS_PATH)
    logging.info("Model metrics saved successfully.")


def test_saved_models() -> None:
    """Test saved models with sample inputs."""

    print("\nTesting Saved Models...")

    ticket_model = joblib.load(TICKET_MODEL_PATH)
    repair_model = joblib.load(REPAIR_MODEL_PATH)

    sample_complaint = ["Battery drains very fast and phone heats while charging"]
    predicted_type = ticket_model.predict(sample_complaint)[0]

    print("\nSample Complaint:", sample_complaint[0])
    print("Predicted Complaint Type:", predicted_type)

    sample_repair_data = pd.DataFrame(
        [
            {
                "Complaint_Type": predicted_type,
                "Brand": "Samsung",
                "Priority": "High",
                "Warranty_Status": "Out of Warranty",
                "Parts_Replaced": "Battery",
                "Repair_Cost": 2500,
            }
        ]
    )

    predicted_days = repair_model.predict(sample_repair_data)[0]

    print("Predicted Repair Time:", round(predicted_days, 2), "days")


def main() -> None:
    """Main training function."""

    setup_logging()
    create_directories()

    try:
        df = load_data()

        ticket_model, ticket_result = train_ticket_classifier(df)
        repair_model, repair_result = train_repair_time_model(df)

        save_metrics(ticket_result, repair_result)
        test_saved_models()

        print("\nTraining completed successfully!")
        print("\nGenerated files:")
        print("1. models/ticket_classifier.pkl")
        print("2. models/repair_prediction.pkl")
        print("3. models/tfidf.pkl")
        print("4. models/classification_report.txt")
        print("5. models/model_metrics.csv")
        print("6. logs/training.log")

    except Exception as error:
        logging.exception("Training failed.")
        print("\nError occurred during training:")
        print(error)


if __name__ == "__main__":
    main()