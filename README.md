# 📊 Text Mining 2025 – Financial Sentiment Classification

This repository presents the final project for the **Text Mining** course of the Master’s in Data Science and Advanced Analytics – NOVA IMS.

## 🧠 Project Objective

The goal is to classify financial tweets into three sentiment categories: **Bearish**, **Bullish**, and **Neutral**, leveraging Natural Language Processing (NLP) techniques. Tweets are noisy by nature and reflect informal language, posing unique challenges for sentiment analysis.

## 🗂️ Repository Structure
```
text_mining_2025/
├── Dados/ # Raw datasets (excluded from Git)
├── Entrega Final/ # Final notebooks and predictions
│ ├── pred_11.csv # Final predictions on test set
│ ├── tm_final_11.ipynb # Clean final notebook
│ └── tm_testes_11.ipynb # Model experimentation
├── Inês & Luis & Pedro & Rafael & Rodrigo/ # Individual contributions
├── Project Handout TM 2025.pdf # Official project handout
└── README.md # Project overview and structure
```

## 🛠️ Key Components

- **EDA & Preprocessing**: Exploration of target imbalance, stopword patterns, and character distributions. Cleaning strategies were adapted for different model families.
- **Modeling Approaches**:
  - Classical models (BoW + KNN, TF-IDF + SVM)
  - Neural Networks (Word2Vec + CNN-BiLSTM, GloVe variants)
  - Transformer-based encoders (RoBERTa, DeBERTa, BERTweet)
- **Model Evaluation**: Metrics include accuracy and macro F1-score. RoBERTa-Large was selected as the final model after fine-tuning with Optuna.
- **Streamlit App**: A lightweight app was built to test the final model in real-time and perform batch evaluations.  
  👉 [Try it here](https://textminingproject.streamlit.app/)

## 🚀 Results

- **Best Transformer**: RoBERTa-Large
- **Validation Metrics**:  
  - Accuracy: **91%**  
  - Macro F1-Score: **0.89**

## 🔑 Keywords

NLP · Sentiment Analysis · Financial Texts · Transformer Models · RoBERTa · Text Classification · Streamlit · BERT · Tokenization
