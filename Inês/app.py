# app.py

import os
import re
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from dotenv import load_dotenv
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from openai import AzureOpenAI

# 1) Load environment variables from .env
load_dotenv()  # expects AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_KEY, AZURE_OPENAI_API_VERSION, AZURE_OPENAI_DEPLOYMENT

AZURE_OPENAI_ENDPOINT    = os.getenv("AZURE_OPENAI_ENDPOINT", "").rstrip("/")
AZURE_OPENAI_KEY         = os.getenv("AZURE_OPENAI_KEY", "")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
AZURE_OPENAI_DEPLOYMENT  = os.getenv("AZURE_OPENAI_DEPLOYMENT", "ChatGPT")

if not (AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_KEY):
    st.error("❌ Please set AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_KEY in your .env file")
    st.stop()

# 2) Initialize AzureOpenAI client
client = AzureOpenAI(
    api_key        = AZURE_OPENAI_KEY,
    azure_endpoint = AZURE_OPENAI_ENDPOINT,
    api_version    = AZURE_OPENAI_API_VERSION,
)

# 3) Define valid labels and mapping
VALID_LABELS = ["Bullish", "Bearish", "Neutral"]
LABEL_MAP    = {0: "Bearish", 1: "Bullish", 2: "Neutral"}

# 4) Helper to clean raw model output
def clean_label(raw: str) -> str:
    m = re.search(r"\b(Bullish|Bearish|Neutral)\b", raw, re.IGNORECASE)
    return m.group(1).capitalize() if m else "Unknown"

# 5) Streamlit page config
st.set_page_config(
    page_title="📈 Stock Sentiment Predictor",
    page_icon="📊",
    layout="wide"
)
st.title("📈 Stock Sentiment Predictor")

# 6) Create tabs
tabs = st.tabs([
    "🔮 Single Classification",
    "📊 Batch Evaluation",
    "👥 Meet the Team"
])

# ————— Tab 1: Single Classification —————
with tabs[0]:
    st.header("Classify a Single Text")
    user_text = st.text_area("Enter a text to analyze:", height=150)

    if st.button("Analyze Sentiment", key="single"):
        if not user_text.strip():
            st.warning("Please enter some text first!")
        else:
            with st.spinner("Calling the model…"):
                try:
                    response = client.chat.completions.create(
                        model=AZURE_OPENAI_DEPLOYMENT,
                        messages=[
                            {
                                "role": "system",
                                "content": (
                                    "You are a stock-market sentiment classifier. "
                                    "Reply only with one of: Bullish, Bearish, or Neutral. "
                                    "Do not include any extra text or explanation."
                                )
                            },
                            {"role": "user", "content": user_text}
                        ],
                        temperature=0.2,
                        top_p=0.9
                    )
                    raw = response.choices[0].message.content.strip()
                    label = clean_label(raw)
                    st.success(f"**Sentiment:** {label}")
                except Exception as e:
                    st.error(f"API call failed: {e}")

# ————— Tab 2: Batch Evaluation —————
with tabs[1]:
    st.header("Batch Evaluation")
    st.write(
        "Upload a CSV file with columns **`text`** and **`label`**, "
        "where `label` is 0 (Bearish), 1 (Bullish) or 2 (Neutral)."
    )
    upload = st.file_uploader("Choose CSV", type=["csv"], key="batch")

    if upload is not None:
        df = pd.read_csv(upload)
        if not {"text", "label"}.issubset(df.columns):
            st.error("CSV must contain `text` and `label` columns.")
        else:
            df["true_label"] = df["label"].astype(int).map(LABEL_MAP)
            texts      = df["text"].astype(str).tolist()
            y_true_all = df["true_label"].tolist()

            y_pred, y_true, skipped = [], [], 0

            with st.spinner("Generating predictions…"):
                for txt, true_lbl in zip(texts, y_true_all):
                    try:
                        resp = client.chat.completions.create(
                            model=AZURE_OPENAI_DEPLOYMENT,
                            messages=[
                                {"role": "system", "content":
                                    "You are a stock-market sentiment classifier. "
                                    "Reply only with one of: Bullish, Bearish, or Neutral."},
                                {"role": "user",   "content": txt}
                            ],
                            temperature=0.2,
                            top_p=0.9
                        )
                        raw  = resp.choices[0].message.content.strip()
                        pred = clean_label(raw)
                        if pred in VALID_LABELS:
                            y_pred.append(pred)
                            y_true.append(true_lbl)
                        else:
                            skipped += 1

                    except Exception as e:
                        err = str(e).lower()
                        if "policy" in err or "filtered" in err:
                            skipped += 1
                        else:
                            raise

            if skipped:
                st.warning(f"⚠️ {skipped} texts were skipped (policy filter or invalid label).")

            if y_true:
                acc    = accuracy_score(y_true, y_pred)
                report = classification_report(y_true, y_pred, output_dict=True)
                cm     = confusion_matrix(y_true, y_pred, labels=VALID_LABELS)
                report_df = pd.DataFrame(report).T

                st.subheader("📊 Accuracy")
                st.write(f"**{acc:.2%}**")

                st.subheader("📝 Classification Report")
                st.dataframe(report_df, use_container_width=True)

                st.subheader("🔢 Confusion Matrix")
                fig, ax = plt.subplots(figsize=(4,4))
                im = ax.imshow(cm, interpolation="nearest")
                ax.set_xticks(range(len(VALID_LABELS)))
                ax.set_yticks(range(len(VALID_LABELS)))
                ax.set_xticklabels(VALID_LABELS, rotation=45)
                ax.set_yticklabels(VALID_LABELS)
                ax.set_xlabel("Predicted")
                ax.set_ylabel("True")
                for i in range(len(VALID_LABELS)):
                    for j in range(len(VALID_LABELS)):
                        ax.text(j, i, cm[i,j], ha="center", va="center")
                st.pyplot(fig, use_container_width=False)

                # build CSV for download
                header_line = f"Skipped texts: {skipped}\n\n"
                metrics_csv = report_df.to_csv()
                cm_df       = pd.DataFrame(cm, index=VALID_LABELS, columns=VALID_LABELS)
                cm_csv      = cm_df.to_csv()
                combined    = (
                    header_line +
                    "Classification Report\n" +
                    metrics_csv +
                    "\nConfusion Matrix\n" +
                    cm_csv
                )
                st.download_button(
                    label="📥 Download metrics & confusion matrix as CSV",
                    data=combined,
                    file_name="model_evaluation.csv",
                    mime="text/csv"
                )
            else:
                st.error("No valid predictions to evaluate after filtering.")

# Tab 3: Meet the Team
with tabs[2]:
    st.header("👥 Meet the Team")
    st.write("Here’s the amazing team behind this project:")

    # Define your team members, now including a GitHub URL
    team = [
        {
            "name": "Inês",
            "file": "inesphoto.jpeg",
            "bio": "Data Scientist and Evolutionary Learning enthusiast.",
            "git": "https://github.com/inesnmajor"
        },
        {
            "name": "Luis",
            "file": "luisphoto.jpeg",
            "bio": "Data Scientist and Product Manager enthusiast",
            "git": "https://github.com/luispsDev"
        },
        {
            "name": "Pedro",
            "file": "pedrophoto.jpeg",
            "bio": "Data Scientist and ML/DL enthusiast",
            "git": "https://github.com/Pedro-P-Santos"
        },
        {
            "name": "Rafael",
            "file": "rafaelphoto.jpeg",
            "bio": "Data Scientist and Customer Behavior Analyst enthusiast.",
            "git": "https://github.com/rafabernard0"
        },
        {
            "name": "Rodrigo",
            "file": "rodrigophoto.jpeg",
            "bio": "Data Scientist and NLP enthusiast",
            "git": "https://github.com/rmiranda37"
        },
    ]

    cols = st.columns(len(team))
    for col, member in zip(cols, team):
        img_path = os.path.join(os.path.dirname(__file__), member["file"])
        col.image(img_path, width=150)
        col.subheader(member["name"])
        col.write(member["bio"])
        col.markdown(
            f'<a href="{member["git"]}" style="text-decoration: none; color: #1f77b4;">🔗 GitHub</a>',
            unsafe_allow_html=True
        )

