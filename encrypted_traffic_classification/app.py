from pathlib import Path
import json
import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).parent
DATA = ROOT / "data" / "synthetic_network_flows.csv"
MODEL_DIR = ROOT / "models"
FEATURES = [
    "flow_duration_ms", "fwd_packets", "bwd_packets", "fwd_bytes", "bwd_bytes",
    "packet_rate", "mean_interarrival_ms", "destination_port", "syn_flag_ratio",
    "bytes_per_packet"
]

st.set_page_config(page_title="Encrypted Flow Lens", page_icon="🛡️", layout="wide")
st.title("🛡️ Encrypted Flow Lens")
st.caption("Network-flow classification without inspecting packet payloads")
st.warning("Portfolio prototype: the bundled dataset is synthetic. Predictions demonstrate a workflow, not validated real-world threat detection.")

if not (MODEL_DIR / "random_forest.joblib").exists():
    st.error("Model files not found. In the project folder, run `python generate_data.py` then `python train_model.py`.")
    st.stop()

model_choice = st.selectbox("Model", ["random_forest", "logistic_regression"])
model = joblib.load(MODEL_DIR / f"{model_choice}.joblib")

tab1, tab2, tab3 = st.tabs(["Dataset explorer", "Classify a flow", "Model evaluation"])
with tab1:
    if DATA.exists():
        df = pd.read_csv(DATA)
        c1, c2, c3 = st.columns(3)
        c1.metric("Synthetic flows", f"{len(df):,}")
        c2.metric("Flow features", len(FEATURES))
        c3.metric("Classes", df["label"].nunique())
        st.bar_chart(df["label"].value_counts())
        st.dataframe(df.head(100), use_container_width=True)
        st.download_button("Download dataset CSV", df.to_csv(index=False).encode("utf-8"), "synthetic_network_flows.csv", "text/csv")
    else:
        st.info("Run `python generate_data.py` to create the demo dataset.")

with tab2:
    st.write("Enter metadata/features from a network flow. No payload content is requested or inspected.")
    with st.form("flow_form"):
        a, b, c = st.columns(3)
        duration = a.number_input("Flow duration (ms)", min_value=0.0, value=1500.0)
        fwd_packets = b.number_input("Forward packets", min_value=0, value=20, step=1)
        bwd_packets = c.number_input("Backward packets", min_value=0, value=15, step=1)
        fwd_bytes = a.number_input("Forward bytes", min_value=0.0, value=12000.0)
        bwd_bytes = b.number_input("Backward bytes", min_value=0.0, value=9000.0)
        packet_rate = c.number_input("Packets per second (feature)", min_value=0.0, value=23.0)
        mean_iat = a.number_input("Mean inter-arrival time (ms)", min_value=0.0, value=20.0)
        port = b.number_input("Destination port", min_value=0, max_value=65535, value=443, step=1)
        syn_ratio = c.slider("SYN flag ratio", 0.0, 1.0, 0.2)
        bytes_per_packet = a.number_input("Bytes per packet", min_value=0.0, value=600.0)
        submitted = st.form_submit_button("Classify flow")
    if submitted:
        sample = pd.DataFrame([{
            "flow_duration_ms": duration, "fwd_packets": fwd_packets, "bwd_packets": bwd_packets,
            "fwd_bytes": fwd_bytes, "bwd_bytes": bwd_bytes, "packet_rate": packet_rate,
            "mean_interarrival_ms": mean_iat, "destination_port": port,
            "syn_flag_ratio": syn_ratio, "bytes_per_packet": bytes_per_packet
        }], columns=FEATURES)
        prediction = model.predict(sample)[0]
        st.subheader(f"Predicted class: {prediction.upper()}")
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(sample)[0]
            st.write("Model scores (not calibrated threat probabilities)")
            st.dataframe(pd.DataFrame({"class": model.classes_, "score": probabilities}).sort_values("score", ascending=False), use_container_width=True)
        st.caption("Treat this as a triage hint only. Confirm suspicious activity with real telemetry and analyst review.")

with tab3:
    metrics_file = MODEL_DIR / "metrics.json"
    if metrics_file.exists():
        metrics = json.loads(metrics_file.read_text(encoding="utf-8"))
        chosen = metrics[model_choice]
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Accuracy", f"{chosen['accuracy']:.1%}")
        m2.metric("Macro precision", f"{chosen['precision_macro']:.1%}")
        m3.metric("Macro recall", f"{chosen['recall_macro']:.1%}")
        m4.metric("Macro F1", f"{chosen['f1_macro']:.1%}")
        labels = chosen["confusion_matrix_labels"]
        cm = pd.DataFrame(chosen["confusion_matrix"], index=[f"Actual {x}" for x in labels], columns=[f"Predicted {x}" for x in labels])
        st.write("Confusion matrix")
        st.dataframe(cm, use_container_width=True)
        st.json(chosen["classification_report"])
        st.caption("Metrics are calculated on a held-out split of synthetic data. They should not be presented as real-world performance.")
    else:
        st.info("Run `python train_model.py` to create evaluation metrics.")
