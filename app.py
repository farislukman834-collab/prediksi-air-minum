import streamlit as st
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import time

# 1. Konfigurasi Halaman (Premium & Wide)
st.set_page_config(page_title="Water Quality AI", page_icon="💧", layout="wide")

# Custom CSS untuk mempercantik elemen visual
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stButton>button { width: 100%; border-radius: 8px; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

# 2. Load Data & Latih Model (Menggunakan Cache agar Cepat)
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

# 3. Sidebar Menu Informasi Model
with st.sidebar:
    st.image("https://img.icons8.com/fluent/96/000000/water-droplet.png", width=80)
    st.header("⚙️ Sistem Engine AI")
    st.metric(label="Akurasi Model Penguji", value=f"{akurasi:.2f}%")
    st.info("Aplikasi ini menggunakan algoritma **Gradient Boosting Classifier** untuk menganalisis parameter kimia air secara real-time.")
    st.divider()
    st.caption("Dikembangkan untuk Analisis Kualitas Air Minum Lab Eksperimental v2.5")

# 4. Header Utama Dashboard
st.title("💧 Smart Water Analytics Dashboard")
st.write("Sistem Cerdas Pendeteksi Kelayakan Air Minum Berbasis Komputasi Awan (Cloud)")
st.divider()

# 5. Form Input Menggunakan Fitur TABS (Lebih Rapi & Ringkas)
st.markdown("### 📊 Pengaturan Parameter Sampel Air")
tab1, tab2 = st.tabs(["🧪 Parameter Kimia Inti", "🔬 Parameter Fisika & Tambahan"])

with tab1:
    col1, col2 = st.columns(2)
    with col1:
        ph = st.slider("Tingkat Keasaman (pH)", 0.0, 14.0, 7.0, 0.1)
        hardness = st.number_input("Kekerasan Air (Hardness - mg/L)", min_value=0.0, value=180.0)
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
        trihalomethanes = st.number_input("Kandungan Trihalometana (ppm)", min_value=0.0, value=60.0)
        
    st.divider()
    turbidity = st.slider("Tingkat Kekeruhan Air (Turbidity - NTU)", 0.0, 10.0, 4.0, 0.1)

# 6. Analisis Status pH Instan (Real-time Feedback)
if ph < 6.5:
    st.warning("⚠️ **Peringatan Lab:** Kondisi sampel air saat ini terlalu **Asam** (Di bawah standar normal 6.5 - 8.5).")
elif ph > 8.5:
    st.warning("⚠️ **Peringatan Lab:** Kondisi sampel air saat ini terlalu **Basa** (Di atas standar normal 6.5 - 8.5).")
else:
    st.success("💡 **Catatan Lab:** Tingkat keasaman (pH) berada di batas ideal **Normal/Netral**.")

st.markdown("<br>", unsafe_allow_html=True)

# 7. Tombol Prediksi dengan Animasi Loading Bar
if st.button("🚀 JALANKAN ANALISIS PREDIKSI AI", type="primary"):
    # Animasi Loading
    with st.spinner("Mengomparasi parameter laboratorium dengan database AI..."):
        time.sleep(1.2) # Efek delay dramatis selama 1.2 detik
    
    # Mempersiapkan data untuk model
    data_input = pd.DataFrame([[ph, hardness, solids, chloramines, sulfate, conductivity, organic_carbon, trihalomethanes, turbidity]],
                              columns=['ph', 'Hardness', 'Solids', 'Chloramines', 'Sulfate', 'Conductivity', 'Organic_carbon', 'Trihalomethanes', 'Turbidity'])
    data_input = data_input[model_ai.feature_names_in_]
    
    # Prediksi
    hasil_prediksi = model_ai.predict(data_input)[0]

    st.markdown("### 📋 Laporan Hasil Prediksi Akhir")
    
    # Tampilan Output yang Dipercantik menggunakan Layout Kolom Besar
    res_col1, res_col2 = st.columns([1, 2])
    
    with res_col1:
        if hasil_prediksi == 1:
            st.metric(label="STATUS KELAYAKAN", value="LAYAK MINUM (POTABLE)")
        else:
            st.metric(label="STATUS KELAYAKAN", value="TIDAK LAYAK (NON-POTABLE)")

    with res_col2:
        if hasil_prediksi == 1:
            st.success("✨ **REKOMENDASI:** Sampel air ini aman dan memenuhi standar baku mutu kesehatan untuk dikonsumsi langsung.")
            st.balloons()
        else:
            st.error("❌ **REKOMENDASI:** Kualitas kimia air berbahaya! Dilarang keras meminum sampel air ini sebelum melalui proses filtrasi, netralisasi pH, atau perebusan intensif.")
