import streamlit as st
import joblib
import pandas as pd

# Load model dan LabelEncoder
model = joblib.load("model_diabet_rfc.joblib")
le = joblib.load("label_encoder.joblib")

st.title("📊 Aplikasi Prediksi Tingkat Obesitas")
st.markdown("Masukkan data untuk memprediksi kategori berat badan:")

with st.form("input_form"):
    # Input numerik
    age = st.number_input("Usia (Tahun)", min_value=1, max_value=100, value=25)
    weight = st.number_input("Berat Badan (kg)", min_value=30.0, max_value=200.0, value=65.0)
    ch2o = st.slider("Konsumsi Air Harian (liter)", 1.0, 3.0, 2.0)
    faf = st.slider("Frekuensi Aktivitas Fisik (skala 0-3)", 0.0, 3.0, 1.0)
    tue = st.slider("Waktu Penggunaan Elektronik (jam/hari)", 0.0, 24.0, 2.0)
    
    # Tombol submit
    submitted = st.form_submit_button("Prediksi")

if submitted:
    # Buat DataFrame dari input
    input_data = {
        'Age': age,
        'Weight': weight,
        'CH2O': ch2o,
        'FAF': faf,
        'TUE': tue
    }
    
    df = pd.DataFrame([input_data])
    
    try:
        # Lakukan prediksi
        prediction = model.predict(df)
        proba = model.predict_proba(df)
        
        # Decode prediction
        prediction_label = le.inverse_transform(prediction)[0]
        
        # Tampilkan hasil
        st.subheader("🩺 Hasil Prediksi")
        st.success(f"Kategori Berat Badan: **{prediction_label}**")
        
    except Exception as e:
        st.error(f"Terjadi error: {str(e)}")

st.markdown("---")
st.info("""
**Keterangan Fitur:**
- **FAF (Frekuensi Aktivitas Fisik):**
  - 0: Tidak pernah
  - 1: 1-2 hari/minggu
  - 2: 3-4 hari/minggu
  - 3: Setiap hari
- **NObeyesdad Kategori:**
  - Insufficient_Weight : Berat badan di bawah normal.
  - Normal_Weight : Berat badan ideal.
  - Overweight_Level_I : Kelebihan berat badan tahap awal.
  - Overweight_Level_II : Mendekati ambang obesitas.
  - Obesity_Type_I : Obesitas ringan dengan risiko kesehatan.
  - Obesity_Type_II : Obesitas sedang dan komplikasi serius. 
  - Obesity_Type_III : Obesitas parah dengan risiko mengancam nyawa.
""")
