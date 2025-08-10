import json
import os
import re
import string
import gdown
import streamlit as st
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification, BertForSequenceClassification, BertConfig
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import nltk

nltk.download('punkt')
nltk.download('stopwords')

# ======= LINK MODEL GOOGLE DRIVE (GANTI SESUAI FILE MODELMU) =======
MODEL_FOLDER = "model_finetuned"
MODEL_FILES = {
    "config.json": "https://drive.google.com/uc?id=1a4Yl35yHFOKPHNcEIynko2cDHfHqQ9PR",
    "model.safetensors": "https://drive.google.com/uc?id=16xGLQkVUwEkhCbL_QYJQSIOMyvpdWuiS",
    "special_tokens_map.json": "https://drive.google.com/uc?id=1n0Sk8pmgYYZtTGXPRbTvN7GXGo8YILeB",
    "tokenizer_config.json": "https://drive.google.com/uc?id=10tw-9e5BP7uHp6Gxlb_BVSGqLQGjNWWK",
    "vocab.txt": "https://drive.google.com/uc?id=1vAtkdbOCYU4QUcj44K9nDaQ7rsnjA86I"
}

# ======= Fungsi download model dari Drive =======
def download_model():
    if not os.path.exists(MODEL_FOLDER):
        os.makedirs(MODEL_FOLDER)
    for filename, url in MODEL_FILES.items():
        path = os.path.join(MODEL_FOLDER, filename)
        if not os.path.exists(path):
            with st.spinner(f"Mengunduh {filename}..."):
                gdown.download(url, path, quiet=False)

# ======= Preprocessing sesuai skripsi =======
factory = StemmerFactory()
stemmer = factory.create_stemmer()
stop_words = stopwords.words('indonesian')
stop_words.extend(['amp', 'nih', 'sih', 'nya', 'gue', 'kalo', 'deh', 'ya'])

def cleaning_data(text):
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'bit.ly/\S+', '', text)
    text = re.sub(r'#\S+', '', text)
    text = re.sub(r'@\S+', '', text)
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'[^\x00-\x7f]', '', text)
    text = text.replace('\n', ' ')
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def lower_case(text):
    return text.lower()

def filter_stopword(text):
    word_tokens = word_tokenize(text)
    filtered = [w for w in word_tokens if w not in stop_words or w in ["satu", "dua", "tiga", "empat", "lima", "enam", "tujuh", "delapan", "sembilan"]]
    return " ".join(filtered)

def stemming_data(text):
    return stemmer.stem(text)

def preprocess_text(text):
    text = cleaning_data(text)
    text = lower_case(text)
    text = filter_stopword(text)
    text = stemming_data(text)
    return text

# ======= Load model & tokenizer =======
@st.cache_resource
def load_model_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_FOLDER)
    config = AutoConfig.from_pretrained(MODEL_FOLDER)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_FOLDER, config=config)
    model.eval()
    return tokenizer, model

# ======= Prediksi sentimen =======
def predict_sentiment(texts, tokenizer, model):
    inputs = tokenizer(texts, return_tensors='pt', padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    preds = torch.argmax(outputs.logits, axis=1).cpu().numpy()
    label_map = {0: "Negatif", 1: "Positif"}
    return [label_map[p] for p in preds]

# ======= Streamlit App =======
def main():
    st.set_page_config(page_title="Prediksi Sentimen Impor", layout="wide")
    st.title("🧵 Aplikasi Prediksi Sentimen Berita Impor Tekstil")
    st.write("Upload file CSV yang berisi kolom `Isi` berisi teks berita impor tekstil.")

    download_model()

    tokenizer, model = load_model_tokenizer()

    uploaded_file = st.file_uploader("Upload file CSV", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        if 'Isi' not in df.columns:
            st.error("File CSV harus memiliki kolom 'Isi'")
            return

        with st.spinner("Memproses data dan melakukan prediksi..."):
            df['CleanNews'] = df['Isi'].astype(str).apply(preprocess_text)
            preds = predict_sentiment(df['CleanNews'].tolist(), tokenizer, model)
            df['sentimen'] = preds

        st.success("Prediksi selesai!")
        st.dataframe(df[['Isi', 'sentimen']])

        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("Download hasil prediksi CSV", data=csv, file_name="hasil_prediksi.csv", mime="text/csv")

if __name__ == "__main__":
    main()
