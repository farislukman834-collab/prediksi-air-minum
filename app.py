import streamlit as st
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import time

# 1. Konfigurasi Halaman (Premium & Wide)
st.set_page_config(page_title="Prediksi Kualitas Air Minum", page_icon="💧", layout="wide")

# 2. INJEKSI CSS KUSTOM (Tema Cerah Premium + Animasi Bergerak)
st.markdown("""
    <style>
    /* Latar belakang cerah dengan gradasi lembut, tidak putih polos */
    .stApp {
        background: linear-gradient(135deg, #eef2f3 0%, #8e9eab 100%);
        color: #2c3e50;
    }
    
    /* ANIMASI BERGERAK PADA JUDUL (Animated Gradient) */
    @keyframes gradient-animation {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    .animated-title {
        font-size: 2.8rem !important;
        font-weight: 800 !important;
        background: linear-gradient(-45deg, #00c6ff, #0072ff, #3a7bd5, #3a6073);
        background-size: 400% 400%;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: gradient-animation 8s ease infinite;
        margin-bottom: 5px;
    }
    
    /* Kustomisasi Sidebar Cerah */
    [data-testid="stSidebar"] {
        background-color: #ffffff !important;
        border-right: 2px solid #dcdde1;
    }
    
    /* Desain Tab Cerah Modern */
    .stTabs [data-baseweb="tab"] {
        color: #7f8c8d !important;
        font-weight: 600;
        font-size: 1.05rem;
    }
    .stTabs [aria-selected="true"] {
        color: #0072ff !important;
        border-bottom-color: #0072ff !important;
    }
    
    /* Tombol Prediksi Interaktif dengan Efek Transisi Glow */
    .stButton>button {
        background: linear-gradient(90deg, #0072ff, #00c6ff) !important;
        color: white !important;
        border: none !important;
        padding: 14px 28px !important;
        font-size: 1.1rem !important;
        font-weight: bold !important;
        border-radius: 12px !important;
        box-shadow: 0 4px 15px rgba(0, 114, 255, 0.3);
        transition: all 0.3s ease-in-out;
    }
    .stButton>button:hover {
        transform: scale(1.02);
        box-shadow: 0 6px 25px rgba(0, 114, 255, 0.5);
    }
    
    /* Kartu Hasil Kelayakan (Cerah & Berbayang Lembut) */
    .card-success {
        background-color: #ffffff;
        border-left: 8px solid #2ecc71;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 8px 24px rgba(46, 204, 113, 0.15);
        margin-top: 15px;
    }
    
    .card-danger {
        background-color: #ffffff;
        border-left: 8px solid #e74c3c;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 8px 24px rgba(231, 76, 60, 0.15);
        margin-top: 15px;
    }
    </style>
    """, unsafe_allow_html=True)

# 3. Load Data & Latih Model AI (Cached)
@st.cache_resource
def latih_model_baru():
    data_air = pd.read_csv('water_potability.csv').dropna()
    X = data_air.drop(columns=['Potability'])
    y = data_air['Potability']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
    model.fit(X_train, y_train)

    pred_test = model.predict(X_test)
    skor_akurasi = accuracy_score(y_test, pred_test)

    return model, skor_akurasi * 100

model_ai, akurasi = latih_model_baru()

# 4. Sidebar Menu Cerah & Bersih
with st.sidebar:
    st.markdown("<br>", unsafe_allow_html=True)
    st.image("https://img.icons8.com/fluent/96/000000/water-droplet.png", width=70)
    st.markdown("### ⚙️ Engine AI")
    st.metric(label="Akurasi Model Penguji", value=f"{akurasi:.2f}%")
    st.divider()
    st.info("Model dilatih menggunakan metode otomatis berbasis algoritma **Gradient Boosting**.")
    st.caption("Lab Eksperimental Kualitas Air v3.5")

# 5. Header Utama dengan Animasi Judul Berjalan
st.markdown('<h1 class="animated-title">💧 Prediksi Kualitas Air Minum</h1>', unsafe_allow_html=True)
st.markdown("<p style='font-size:1.1rem; color:#34495e;'>Sistem Cerdas Analisis Baku Mutu dan Kelayakan Sampel Air Laboratorium</p>", unsafe_allow_html=True)
st.divider()

# 6. Kontainer Input Parameter (Dibagi dalam
