import streamlit as st
import numpy as np
from PIL import Image, ImageOps
import tensorflow as tf

# Judul Aplikasi
st.title("Aplikasi Deteksi Sampah (Organik & Anorganik)")
st.write("Arahkan kamera atau upload foto sampah untuk mendeteksi jenisnya.")

# Load Model TFLite dan Labels
@st.cache_resource
def load_model_and_labels():
    # Memuat model TFLite
    interpreter = tf.lite.Interpreter(model_path="model.tflite")
    interpreter.allocate_tensors()
    
    # Memuat label dari file labels.txt (pastikan barisnya murni teks: organik\nanorganik)
    with open("labels.txt", "r") as f:
        labels = [line.strip() for line in f.readlines() if line.strip()]
        
    return interpreter, labels

try:
    interpreter, labels = load_model_and_labels()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
except Exception as e:
    st.error(f"Gagal memuat model atau file label: {e}")
    st.stop()

# Ukuran input standar Teachable Machine (biasanya 224x224)
H, W = input_details[0]['shape'][1], input_details[0]['shape'][2]

def preprocess_image(img: Image.Image):
    # Pastikan mode warna RGB
    img = img.convert("RGB")
    # Resize dengan mempertahankan rasio
    img = ImageOps.fit(img, (W, H), Image.Resampling.LANCZOS)
    arr = np.array(img)
    
    # Menyesuaikan tipe data input model (uint8 atau float32)
    input_dtype = input_details[0]['dtype']
    if input_dtype == np.float32:
        arr = arr.astype(np.float32) / 255.0
    else:
        arr = arr.astype(input_dtype)
        
    # Tambahkan dimensi batch [1, H, W, C]
    arr = np.expand_dims(arr, axis=0)
    return arr

# Pilihan Input Gambar (Kamera atau Upload)
option = st.radio("Pilih metode input:", ("Upload Foto", "Gunakan Kamera"))

image = None
if option == "Upload Foto":
    uploaded_file = st.file_uploader("Pilih gambar sampah...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
else:
    camera_file = st.camera_input("Ambil foto sampah")
    if camera_file is not None:
        image = Image.open(camera_file)

# Proses Prediksi jika gambar tersedia
if image is not None:
    st.image(image, caption="Gambar yang Diuji", use_column_width=True)
    st.write("Mendeteksi...")
    
    # Pra-pemrosesan gambar
    input_data = preprocess_image(image)
    
    # Jalankan inferensi model TFLite
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])[0]
    
    # Normalisasi output jika berupa uint8
    if output_details[0]['dtype'] == np.uint8:
        scale, zero_point = output_details[0]['quantization']
        output_data = scale * (output_data.astype(np.float32) - zero_point)
    
    # Ambil hasil prediksi tertinggi
    predicted_index = np.argmax(output_data)
    confidence = float(output_data[predicted_index]) * 100
    
    # Jika model menggunakan output softmax mentah atau logit
    if confidence > 1.0 and output_details[0]['dtype'] != np.uint8:
        # Konversi ke persentase jika rentangnya 0-1
        pass
    
    # Tampilkan Hasil
    st.subheader("Hasil Prediksi:")
    predicted_label = labels[predicted_index] if predicted_index < len(labels) else "Tidak Diketahui"
    
    st.success(f"**Jenis Sampah: {predicted_label.upper()}** (Akurasi: {confidence:.2f}%)")
    
    # Tampilkan rincian probabilitas semua kelas untuk debugging
    with st.expander("Lihat Rincian Probabilitas"):
        for i, lab in enumerate(labels):
            score = float(output_data[i]) * 100 if output_data[i] <= 1.0 else float(output_data[i])
            st.write(f"- {lab.capitalize()}: {score:.2f}%")
            # refresh-app-1
