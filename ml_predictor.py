import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib

DATA_PATH = "data/sample.csv"
MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "student_approval_model.pkl")

def load_data(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset not found at '{path}'. Make sure you created data/sample.csv."
        )
    return pd.read_csv(path)

def train_model(df: pd.DataFrame) -> LogisticRegression:
    required_cols = {"hours_studied", "attendance", "approved"}
    if not required_cols.issubset(df.columns):
        raise ValueError(
            f"CSV must contain columns {required_cols}. Found: {set(df.columns)}"
        )

    X = df[["hours_studied", "attendance"]]
    y = df["approved"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    model = LogisticRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.2f}")

    print("\nClassification report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    return model

def save_model(model: LogisticRegression, model_path: str) -> None:
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)
    print(f"\nModel saved to: {model_path}")

def predict_example(model: LogisticRegression) -> None:
    example = pd.DataFrame({"hours_studied": [4], "attendance": [75]})
    pred = model.predict(example)[0]
    proba = model.predict_proba(example)[0][1]

    label = "APPROVED ✅" if pred == 1 else "NOT APPROVED ❌"
    print("\nExample prediction:")
    print(f"Input: hours_studied=4, attendance=75 -> {label} (prob={proba:.2f})")

def main():
    df = load_data(DATA_PATH)
    model = train_model(df)
    predict_example(model)
    save_model(model, MODEL_PATH)

if __name__ == "__main__":
    main()
