"""Train baseline classifiers on the bundled synthetic flow dataset."""
from pathlib import Path
import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import joblib

ROOT = Path(__file__).parent
DATA = ROOT / "data" / "synthetic_network_flows.csv"
MODEL_DIR = ROOT / "models"
FEATURES = [
    "flow_duration_ms", "fwd_packets", "bwd_packets", "fwd_bytes", "bwd_bytes",
    "packet_rate", "mean_interarrival_ms", "destination_port", "syn_flag_ratio",
    "bytes_per_packet"
]

def main():
    if not DATA.exists():
        raise SystemExit("Dataset missing. Run: python generate_data.py")
    df = pd.read_csv(DATA)
    X = df[FEATURES]
    y = df["label"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    models = {
        "logistic_regression": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42))
        ]),
        "random_forest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("classifier", RandomForestClassifier(n_estimators=250, max_depth=12, class_weight="balanced", random_state=42))
        ])
    }
    MODEL_DIR.mkdir(exist_ok=True)
    metrics = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        metrics[name] = {
            "accuracy": round(float(accuracy_score(y_test, pred)), 4),
            "precision_macro": round(float(precision_score(y_test, pred, average="macro", zero_division=0)), 4),
            "recall_macro": round(float(recall_score(y_test, pred, average="macro", zero_division=0)), 4),
            "f1_macro": round(float(f1_score(y_test, pred, average="macro", zero_division=0)), 4),
            "confusion_matrix_labels": list(model.classes_),
            "confusion_matrix": confusion_matrix(y_test, pred, labels=model.classes_).tolist(),
            "classification_report": classification_report(y_test, pred, output_dict=True, zero_division=0)
        }
        joblib.dump(model, MODEL_DIR / f"{name}.joblib")
        print(f"{name}: {metrics[name]['f1_macro']=} {metrics[name]['accuracy']=}")
    with open(MODEL_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    with open(MODEL_DIR / "feature_columns.json", "w", encoding="utf-8") as f:
        json.dump(FEATURES, f, indent=2)
    print(f"Saved models and metrics to {MODEL_DIR}")

if __name__ == "__main__":
    main()
