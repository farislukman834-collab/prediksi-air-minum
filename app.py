import streamlit as st
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 1. Konfigurasi Halaman & Desain Premium
st.set_page_config(page_title="Prediksi Kualitas Air", page_icon="💧", layout="wide")

st.title("💧 Sistem Cerdas: Prediksi Kualitas Air Minum")
st.caption("Versi 2.0 - Berbasis Algoritma Gradient Boosting & Analisis Parameter Kimia")

# 2. Load Data, Latih Model, & Hitung Akurasi
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

# 3. Panel Dashboard Informasi Model
st.sidebar.header("⚙️ Informasi Sistem AI")
st.sidebar.metric(label="Akurasi Model Penguji", value=f"{akurasi:.2f}%")
st.sidebar.info("Model dilatih menggunakan dataset komprehensif dari Kaggle dengan metode pembersihan data otomatis.")

# 4. Form Input Interaktif dengan Layout 3 Kolom
st.markdown("### 📊 Panel Input Parameter Laboratorium")
col1, col2, col3 = st.columns(3)

with col1:
    ph = st.slider("Tingkat Keasaman (pH)", 0.0, 14.0, 7.0, 0.1)
    hardness = st.number_input("Kekerasan Air (Hardness)", min_value=0.0, value=180.0)
    solids = st.number_input("Total Padatan Terlarut (Solids)", min_value=0.0, value=20000.0)

with col2:
    chloramines = st.number_input("Kandungan Kloramin (Chloramines)", min_value=0.0, value=7.0)
    sulfate = st.number_input("Kandungan Sulfat (Sulfate)", min_value=0.0, value=300.0)
    conductivity = st.number_input("Daya Hantar Listrik (Conductivity)", min_value=0.0, value=400.0)

with col3:
    organic_carbon = st.number_input("Karbon Organik (Organic Carbon)", min_value=0.0, value=15.0)
    trihalomethanes = st.number_input("Kandungan Trihalometana", min_value=0.0, value=60.0)
    turbidity = st.slider("Tingkat Kekeruhan (Turbidity)", 0.0, 10.0, 4.0, 0.1)

# Fitur Tambahan: Analisis Status pH Langsung
st.markdown("---")
if ph < 6.5:
    st.warning("⚠️ Catatan Laboratorium: Sifat air cenderung **Asam** (Di bawah standar normal pH 6.5 - 8.5).")
elif ph > 8.5:
    st.warning("⚠️ Catatan Laboratorium: Sifat air cenderung **Basa** (Di atas standar normal pH 6.5 - 8.5).")
else:
    st.info("💡 Catatan Laboratorium: Tingkat keasaman (pH) berada di batas **Normal/Netral**.")

# 5. Tombol Prediksi AI
if st.button("🚀 Jalankan Analisis AI Sekarang", type="primary"):
    data_input = pd.DataFrame([[ph, hardness, solids, chloramines, sulfate, conductivity, organic_carbon, trihalomethanes, turbidity]],
                              columns=['ph', 'Hardness', 'Solids', 'Chloramines', 'Sulfate', 'Conductivity', 'Organic_carbon', 'Trihalomethanes', 'Turbidity'])

    data_input = data_input[model_ai.feature_names_in_]
    hasil_prediksi = model_ai.predict(data_input)[0]

    st.markdown("### 📋 Hasil Analisis Akhir:")
    if hasil_prediksi == 1:
        st.success("✅ **REKOMENDASI: AIR LAYAK DIKONSUMSI (POTABLE)**")
        st.balloons()
    else:
        st.error("🚨 **REKOMENDASI: AIR TIDAK LAYAK MINUM (NON-POTABLE)**")
        st.warning("Perhatian: Harap lakukan filtrasi atau perebusan ulang karena parameter kimia tidak seimbang.")
