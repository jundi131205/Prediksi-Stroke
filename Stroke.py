import streamlit as st
import joblib
import pandas as pd

# =========================
# Load Model dan Scaler
# =========================
knn_model = joblib.load("knn_model.pkl")
dt_model = joblib.load("decision_tree_model.pkl")
scaler = joblib.load("scaler.pkl")

# =========================
# Konfigurasi Halaman
# =========================
st.set_page_config(
    page_title="Prediksi Risiko Stroke",
    page_icon="🧠",
    layout="centered"
)

st.title("🧠 Prediksi Risiko Stroke")
st.write("""
Aplikasi ini digunakan untuk memprediksi risiko stroke berdasarkan data kesehatan pasien.
Pengguna dapat memilih model **KNN** atau **Decision Tree**, serta melakukan prediksi melalui input manual atau upload file CSV.
""")

# =========================
# Pilih Model
# =========================
st.subheader("Pilih Model Machine Learning")

model_pilihan = st.selectbox(
    "Pilih model yang ingin digunakan:",
    ["KNN", "Decision Tree"]
)

if model_pilihan == "KNN":
    model = knn_model
else:
    model = dt_model

# =========================
# Mapping Encoding
# =========================
gender_map = {"Female": 0, "Male": 1, "Other": 2}
married_map = {"No": 0, "Yes": 1}
work_map = {
    "Govt_job": 0,
    "Never_worked": 1,
    "Private": 2,
    "Self-employed": 3,
    "children": 4
}
residence_map = {"Rural": 0, "Urban": 1}
smoking_map = {
    "Unknown": 0,
    "formerly smoked": 1,
    "never smoked": 2,
    "smokes": 3
}
yes_no_map = {"Tidak": 0, "Ya": 1}

kolom_fitur = [
    "gender",
    "age",
    "hypertension",
    "heart_disease",
    "ever_married",
    "work_type",
    "Residence_type",
    "avg_glucose_level",
    "bmi",
    "smoking_status"
]

# =========================
# Fungsi Preprocessing CSV
# =========================
def preprocess_data(data):
    data = data.copy()

    if "id" in data.columns:
        data = data.drop("id", axis=1)

    if "stroke" in data.columns:
        data = data.drop("stroke", axis=1)

    missing_cols = [col for col in kolom_fitur if col not in data.columns]
    if missing_cols:
        st.error(f"Kolom berikut tidak ditemukan di CSV: {missing_cols}")
        return None

    data = data[kolom_fitur]

    data["gender"] = data["gender"].replace(gender_map)
    data["ever_married"] = data["ever_married"].replace(married_map)
    data["work_type"] = data["work_type"].replace(work_map)
    data["Residence_type"] = data["Residence_type"].replace(residence_map)
    data["smoking_status"] = data["smoking_status"].replace(smoking_map)

    for col in kolom_fitur:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    if data["bmi"].isnull().sum() > 0:
        data["bmi"] = data["bmi"].fillna(data["bmi"].mean())

    if data["bmi"].isnull().sum() > 0:
        data["bmi"] = data["bmi"].fillna(28.89)

    if data.isnull().sum().sum() > 0:
        st.error("Masih terdapat data kosong atau kategori yang tidak sesuai setelah preprocessing.")
        st.write("Cek data berikut:")
        st.dataframe(data[data.isnull().any(axis=1)])
        return None

    return data

# =========================
# Pilih Metode Input
# =========================
st.subheader("Metode Input Data")

metode_input = st.radio(
    "Pilih metode input:",
    ["Input Manual", "Upload CSV"]
)

# =========================
# Input Manual
# =========================
if metode_input == "Input Manual":
    st.subheader("Input Data Pasien")

    gender = st.selectbox("Gender", ["Female", "Male", "Other"])
    age = st.number_input("Usia", min_value=0.0, max_value=120.0, value=30.0)
    hypertension = st.selectbox("Riwayat Hipertensi", ["Tidak", "Ya"])
    heart_disease = st.selectbox("Riwayat Penyakit Jantung", ["Tidak", "Ya"])
    ever_married = st.selectbox("Status Pernikahan", ["No", "Yes"])
    work_type = st.selectbox("Jenis Pekerjaan", ["Private", "Self-employed", "Govt_job", "children", "Never_worked"])
    residence_type = st.selectbox("Tipe Tempat Tinggal", ["Urban", "Rural"])
    avg_glucose_level = st.number_input("Rata-rata Kadar Glukosa", min_value=0.0, value=100.0)
    bmi = st.number_input("BMI", min_value=0.0, value=25.0)
    smoking_status = st.selectbox("Status Merokok", ["formerly smoked", "never smoked", "smokes", "Unknown"])

    if st.button("Prediksi Stroke"):
        input_data = pd.DataFrame([[
            gender_map[gender],
            age,
            yes_no_map[hypertension],
            yes_no_map[heart_disease],
            married_map[ever_married],
            work_map[work_type],
            residence_map[residence_type],
            avg_glucose_level,
            bmi,
            smoking_map[smoking_status]
        ]], columns=kolom_fitur)

        input_scaled = scaler.transform(input_data)
        prediksi = model.predict(input_scaled)

        if prediksi[0] == 1:
            st.error("Hasil Prediksi: Pasien Berisiko Stroke")
        else:
            st.success("Hasil Prediksi: Pasien Tidak Stroke")

# =========================
# Upload CSV
# =========================
else:
    st.subheader("Upload File CSV")

    uploaded_file = st.file_uploader(
        "Upload file CSV dengan kolom yang sesuai dataset stroke",
        type=["csv"]
    )

    st.info("""
    File CSV harus memiliki kolom:
    gender, age, hypertension, heart_disease, ever_married, work_type,
    Residence_type, avg_glucose_level, bmi, smoking_status.

    Kolom id dan stroke boleh ada, tetapi tidak akan ditampilkan pada hasil akhir.
    """)

    if uploaded_file is not None:
        data_csv = pd.read_csv(uploaded_file)

        st.write("Data yang diupload:")
        st.dataframe(data_csv)

        if st.button("Prediksi Data CSV"):
            data_prediksi = preprocess_data(data_csv)

            if data_prediksi is not None:
                data_scaled = scaler.transform(data_prediksi)
                hasil_prediksi = model.predict(data_scaled)

                hasil = data_csv.copy()

                hasil["hasil_prediksi"] = pd.Series(hasil_prediksi).map({
                    0: "Tidak Stroke",
                    1: "Berisiko Stroke"
                })

                if "stroke" in hasil.columns:
                    hasil = hasil.drop("stroke", axis=1)

                if "prediksi" in hasil.columns:
                    hasil = hasil.drop("prediksi", axis=1)

                st.success("Prediksi berhasil dilakukan.")
                st.dataframe(hasil)

                csv_download = hasil.to_csv(index=False).encode("utf-8")

                st.download_button(
                    label="Download Hasil Prediksi",
                    data=csv_download,
                    file_name="hasil_prediksi_stroke.csv",
                    mime="text/csv"
                )

# =========================
# Informasi Model
# =========================
st.markdown("---")
st.subheader("Informasi Model")

if model_pilihan == "KNN":
    st.write("Algoritma: K-Nearest Neighbor (KNN)")
    st.write("Jumlah Tetangga (K):", model.n_neighbors)
    st.write("Metode Jarak:", model.metric)
    st.write("Weights:", model.weights)
else:
    st.write("Algoritma: Decision Tree")
    st.write("Criterion:", model.criterion)
    st.write("Max Depth:", model.max_depth)
    st.write("Min Samples Split:", model.min_samples_split)
    st.write("Min Samples Leaf:", model.min_samples_leaf)

with st.expander("Lihat Detail Model"):
    st.code(str(model))