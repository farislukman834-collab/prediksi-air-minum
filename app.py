import streamlit as st
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import time

# 1. Konfigurasi Halaman (Premium & Wide)
st.set_page_config(page_title="Water Quality AI Pro", page_icon="💧", layout="wide")

# 2. INJEKSI CSS KUSTOM (Mengubah UI agar tidak putih polosan + Efek Glow)
st.markdown("""
    <style>
    /* Mengubah background utama menjadi gradasi gelap elegan */
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        color: #f8fafc;
    }
    
    /* Mempercantik teks judul utama */
    .main-title {
        font-size: 3rem !important;
        font-weight: 800 !important;
        background: linear-gradient(90deg, #38bdf8, #0ea5e9);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0px;
    }
    
    /* Kustomisasi Sidebar */
    [data-testid="stSidebar"] {
        background-color: #0f172a !important;
        border-right: 1px solid #334155;
    }
    
    /* Desain Tab agar kontras */
    .stTabs [data-baseweb="tab"] {
        color: #94a3b8 !important;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        color: #38bdf8 !important;
        border-bottom-color: #38bdf8 !important;
    }
    
    /* Tombol Prediksi Interaktif */
    .stButton>button {
        background: linear-gradient(90deg, #0284c7, #0369a1) !important;
        color: white !important;
        border: none !important;
        padding: 12px 24px !important;
        font-size: 1.1rem !important;
        font-weight: bold !important;
        border-radius: 12px !important;
        box-shadow: 0 4px 12px rgba(2, 132, 199, 0.3);
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(2, 132, 199, 0.5);
    }
    
    /* Efek Glow Card untuk Hasil Positif (Layak) */
    .glow-card-success {
        background: rgba(16, 185, 129, 0.1);
        border: 2px solid #10b981;
        padding: 20px;
        border-radius: 16px;
        box-shadow: 0 0 25px rgba(16, 185, 129, 0.2);
        margin-top: 15px;
    }
    
    /* Efek Glow Card untuk Hasil Negatif (Tidak Layak) */
    .glow-card-danger {
        background: rgba(239, 68, 68, 0.1);
        border: 2px solid #ef4444;
        padding: 20px;
        border-radius: 16px;
        box-shadow: 0 0 25px rgba(239, 68, 68, 0.2);
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

# 4. Sidebar Menu Elegan
with st.sidebar:
    st.markdown("<br>", unsafe_allow_html=True)
    st.image("https://img.icons8.com/fluent/96/000000/water-droplet.png", width=70)
    st.markdown("### ⚙️ ENGINE STATUS")
    st.metric(label="Akurasi Inteligensia", value=f"{akurasi:.2f}%")
    st.markdown("---")
    st.info("Sistem menganalisis kombinasi 9 matriks kimiawi air menggunakan arsitektur **Gradient Boosting Berpohon**.")
    st.caption("Lab Environment Active • v3.0 Neon Edition")

# 5. Header Utama dengan Animasi Warna
st.markdown('<h1 class="main-title">💧 HydroSense AI</h1>', unsafe_allow_html=True)
st.write("Sistem Pemantauan Otomatis Mutu dan Kelayakan Air Konsumsi")
st.markdown("<br>", unsafe_allow_html=True)

# 6. Kontainer Input Parameter (Dibagi dalam Tab Modern)
st.markdown("### 📊 Parameter Hasil Uji Laboratorium")
tab1, tab2 = st.tabs(["🧪 Komposisi Kimia Inti", "🔬 Sifat Fisika & Organik"])

with tab1:
    col1, col2 = st.columns(2)
    with col1:
        ph = st.slider("Derajat Keasaman (pH)", 0.0, 14.0, 7.0, 0.1)
        hardness = st.number_input("Tingkat Kekerasan Air (Hardness - mg/L)", min_value=0.0, value=180.0)
    with col2:
        chloramines = st.number_input("Kadar Kloramin (Chloramines - ppm)", min_value=0.0, value=7.0)
        sulfate = st.number_input("Kandungan Sulfat (Sulfate - mg/L)", min_value=0.0, value=300.0)

with tab2:
    col3, col4 = st.columns(2)
    with col3:
        solids = st.number_input("Total Padatan Terlarut (Solids - ppm)", min_value=0.0, value=20000.0)
        conductivity = st.number_input("Daya Hantar Listrik (Conductivity - μS/cm)", min_value=0.0, value=400.0)
    with col4:
        organic_carbon = st.number_input("Karbon Organik (Organic Carbon - ppm)", min_value=0.0, value=15.0)
        trihalomethanes = st.number_input("Zat Trihalometana (ppm)", min_value=0.0, value=60.0)
        
    st.markdown("<br>", unsafe_allow_html=True)
    turbidity = st.slider("Tingkat Kekeruhan Fisik (Turbidity - NTU)", 0.0, 10.0, 4.0, 0.1)

# Real-time Warning Panel
st.markdown("<br>", unsafe_allow_html=True)
if ph < 6.5:
    st.warning("⚠️ **Indikasi Korosif:** Tingkat pH di bawah standar nasional (6.5 - 8.5). Air bersifat terlalu Asam.")
elif ph > 8.5:
    st.warning("⚠️ **Indikasi Kerak:** Tingkat pH di atas standar nasional (6.5 - 8.5). Air bersifat terlalu Basa/Alkali.")
else:
    st.success("💡 **Parameter Normal:** Tingkat keasaman (pH) berada di rentang ideal aman.")

st.markdown("<br><br>", unsafe_allow_html=True)

# 7. Eksekusi Prediksi & Efek Glow Card
if st.button("🚀 MULAI PEMPROSESAN AI", type="primary"):
    # Efek Loading Progress Bar fiktif agar dramatis
    progress_
