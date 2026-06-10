import streamlit as st
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import time

# 1. Konfigurasi Halaman & Desain Premium Pro
st.set_page_config(page_title="Water Quality AI Pro", page_icon="💧", layout="wide")

# 2. INJEKSI CSS KUSTOM (Tema Premium + Animasi)
st.markdown("""
    <style>
    /* 1. Latar Belakang Gradasi Animasi (Lembut & Elegan) */
    @keyframes gradientBG {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    .stApp {
        background: linear-gradient(-45deg, #e0f2f1, #b2dfdb, #e8f5e9, #b2dfdb);
        background-size: 400% 400%;
        animation: gradientBG 10s ease infinite;
    }
    
    /* 2. Kustomisasi Sidebar */
    [data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #dcdde1;
        border-radius: 0 20px 20px 0;
    }
    
    /* 3. Desain Tabs Modern */
    .stTabs [data-baseweb="tab"] {
        color: #7f8c8d;
        font-weight: 600;
        font-size: 1.1rem;
    }
    .stTabs [aria-selected="true"] {
        color: #00897b;
        border-bottom-color: #00897b;
    }
    
    /* 4. Kustomisasi Judul Utama (Animated Gradient Text) */
    @keyframes gradientTitle {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    .animated-title {
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(-45deg, #00897b, #00c853, #26c6da, #00897b);
        background-size: 300% 300%;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: gradientTitle 5s ease infinite;
        margin-bottom: 0px;
    }
    
    /* 5. Tombol Prediksi Interaktif (Pulsing Effect on Hover) */
    .stButton>button {
        background: linear-gradient(90deg, #00897b, #00c853) !important;
        color: white !important;
        border: none !important;
        padding: 15px 30px !important;
        font-size: 1.2rem !important;
        font-weight: bold !important;
        border-radius: 15px !important;
        box-shadow: 0 4px 15px rgba(0, 137, 123, 0.4);
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(0, 137, 123, 0.6);
    }
    
    /* 6. Kartu Hasil Kelayakan (Animated Success/Danger) */
    @keyframes resultPop {
        0% { transform: scale(0.9); opacity: 0; }
        100% { transform: scale(1); opacity: 1; }
    }
    .result-card-success {
        background: rgba(0, 200, 83, 0.1);
        border: 2px solid #00c853;
        padding: 20px;
        border-radius: 20px;
        color: #00c853;
        font-weight: bold;
        box-shadow: 0 0 20px rgba(0, 200, 83, 0.2);
        animation: resultPop 0.5s ease;
    }
    .result-card-danger {
        background: rgba(229, 57, 53, 0.1);
        border: 2px solid #e53935;
        padding: 20px;
        border-radius: 20px;
        color: #e53935;
        font-weight: bold;
        box-shadow: 0 0 20px rgba(229, 57, 53, 0.2);
        animation: resultPop 0.5s ease;
    }
    
    /* Hilangkan Watermark Streamlit */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

# 3. Load Data, Latih Model, & Hitung Akurasi (Cached)
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

# 4. Sidebar Pro (Modern)
with st.sidebar:
    st.markdown("<br>", unsafe_allow_html=True)
    # Ikon Visual Tetesan Air Ganti URL Ikon
    st.image("https://img.icons8.com/fluent/96/000000/water-droplet.png", width=80)
    st.markdown("### ⚙️ Engine AI Pro")
    st.metric(label="Akurasi Model Penguji", value=f"{akurasi:.2f}%")
    st.divider()
    st.info("Sistem menganalisis 9 matriks kimiawi untuk menentukan kelayakan sampel air secara presisi.")
    st.caption("Lab Eksperimental v4.0 Pro Animated")

# 5. Header Utama (Didesain Kustom)
# st.title("💧 Prediksi Kualitas Air Minum") # Ini standar
st.markdown('<h1 class="animated-title">💧 Prediksi Kualitas Air Minum</h1>', unsafe_allow_html=True)
st.caption("Analisis Baku Mutu Sampel Laboratorium Berbasis Arsitektur Gradient Boosting")
st.divider()

# 6. Panel Input Berbasis TABS (Organized UI)
st.markdown("### 📊 Parameter Hasil Uji Laboratorium")
tab1, tab2 = st.tabs(["🧪 Parameter Kimia Inti", "🔬 Sifat Fisika & Tambahan"])

with tab1:
    col1, col2 = st.columns(2)
    with col1:
        ph = st.slider("Derajat Keasaman (pH)", 0.0, 14.0, 7.0, 0.1)
        hardness = st.number_input("Tingkat Kekerasan Air (Hardness - mg/L)", min_value=0.0, value=180.0)
    with col2:
        chloramines = st.number_input("Kandungan Kloramin (Chloramines - ppm)", min_value=0.0, value=7.0)
        sulfate = st.number_input("Kandungan Sulfat (Sulfate - mg/L)", min_value=0.0, value=300.0)

with tab2:
    col3, col4 = st.columns(2)
    with col3:
        solids = st.number_input("Total Padatan Terlarut (Solids - ppm)", min_value=0.0, value=20000.0)
        conductivity = st.number_input("Daya Hantar Listrik (Conductivity - μS/cm)", min_value=0.0, value=400.0)
    with col4:
        organic_carbon = st.number_input("Karbon Organik (Organic Carbon - ppm)", min_value=0.0, value=15.0)
        trihalomethanes = st.number_input("Zat Trihalometana (ppm)", min_value=0.0, value=60.0)
        
    st.divider()
    turbidity = st.slider("Tingkat Kekeruhan (Turbidity - NTU)", 0.0, 10.0, 4.0, 0.1)

# Fitur Tambahan: Analisis Status pH Langsung (Real-time Feed)
st.markdown("<br>", unsafe_allow_html=True)
if ph < 6.5:
    st.warning("⚠️ **Catatan Laboratorium:** Sifat air cenderung **Asam** (Di bawah standar ideal 6.5 - 8.5).")
elif ph > 8.5:
    st.warning("⚠️ **Catatan Laboratorium:** Sifat air cenderung **Basa** (Di atas standar ideal 6.5 - 8.5).")
else:
    st.success("💡 **Catatan Laboratorium:** Tingkat keasaman (pH) berada di rentang **Ideal/Netral**.")

# 7. Tombol Prediksi Interaktif
st.markdown("<br><br>", unsafe_allow_html=True)
if st.button("🚀 MULAI ANALISIS AI", type="primary"):
    
    # Animasi Loading
    with st.spinner("Sistem sedang memproses parameter laboratorium..."):
        time.sleep(1.5)
        
    # Memproses Data untuk Prediksi
    data_input = pd.DataFrame([[ph, hardness, solids, chloramines, sulfate, conductivity, organic_carbon, trihalomethanes, turbidity]],
                              columns=['ph', 'Hardness', 'Solids', 'Chloramines', 'Sulfate', 'Conductivity', 'Organic_carbon', 'Trihalomethanes', 'Turbidity'])

    data_input = data_input[model_ai.feature_names_in_]
    hasil_prediksi = model_ai.predict(data_input)[0]

    st.divider()
    st.markdown("### 📋 Hasil Laporan Akhir:")
    
    # Tampilan Output yang Dipercantik (Menggunakan HTML Card + Animasi CSS)
    if hasil_prediksi == 1:
        st.markdown("""
            <div class="result-card-success">
                <h2 style='color: #00c853; margin-top:0;'>✨ REKOMENDASI: AIR LAYAK DIKONSUMSI (POTABLE) ✨</h2>
                <p style='color: #2c3e50; font-size: 1.1rem;'>Berdasarkan analisis algoritma cerdas, parameter sampel air berada dalam batas aman baku mutu kesehatan. Air memenuhi syarat untuk dikonsumsi langsung.</p>
            </div>
            """, unsafe_allow_html=True)
        st.balloons()
    else:
        st.markdown("""
            <div class="result-card-danger">
                <h2 style='color: #e53935; margin-top:0;'>❌ REKOMENDASI: AIR BERBAHAYA (NON-POTABLE) ❌</h2>
                <p style='color: #2c3e50; font-size: 1.1rem;'>Perhatian! Parameter kimia terdeteksi di luar ambang batas aman. Dilarang mengonsumsi air ini sebelum dilakukan filtrasi atau penanganan ulang secara intensif.</p>
            </div>
            """, unsafe_allow_html=True)
