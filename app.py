import streamlit as st
import numpy as np
from PIL import Image, ImageOps
import tensorflow as tf

st.title("Aplikasi Deteksi Sampah (Organik & Anorganik)")
st.write("Arahkan kamera atau upload foto sampah untuk mendeteksi jenisnya.")

# Load Model dan Label secara efisien
@st.cache_resource
def load_model_and_labels():
    interpreter = tf.lite.Interpreter(model_path="model.tflite")
    interpreter.allocate_tensors()
    
    # Membaca label dan memastikan tidak ada karakter kosong/angka indeks
    with open("labels.txt", "r") as f:
        labels = [line.strip() for line in f.readlines() if line.strip()]
        # Bersihkan jika masih ada angka di depannya (misal "0 organik" -> "organik")
        labels = [lbl.split(' ', 1)[1] if lbl.split(' ')[0].isdigit() else lbl for lbl in labels]
        
    return interpreter, labels

try:
    interpreter, labels = load_model_and_labels()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
except Exception as e:
    st.error(f"Gagal memuat model: {e}")
    st.stop()

# Ukuran input model Teachable Machine (biasanya 224x224)
H, W = input_details[0]['shape'][1], input_details[0]['shape'][2]

def preprocess_image(img: Image.Image):
    img = img.convert("RGB")
    img = ImageOps.fit(img, (W, H), Image.Resampling.LANCZOS)
    arr = np.array(img)
    
    input_dtype = input_details[0]['dtype']
    
    # Penanganan tipe data input yang tepat untuk TFLite
    if input_dtype == np.float32:
        arr = arr.astype(np.float32) / 255.0
    else:
        arr = arr.astype(input_dtype)
        
    arr = np.expand_dims(arr, axis=0)
    return arr

# Pilihan Input
option = st.radio("Pilih metode input:", ("Upload Foto", "Gunakan Kamera (DroidCam/Webcam)"))

image = None
if option == "Upload Foto":
    uploaded_file = st.file_uploader("Pilih gambar...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
else:
    camera_file = st.camera_input("Ambil foto")
    if camera_file is not None:
        image = Image.open(camera_file)

if image is not None:
    st.image(image, caption="Gambar yang Diuji", use_column_width=True)
    st.write("Mendeteksi...")
    
    # Proses Gambar
    input_data = preprocess_image(image)
    
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])[0]
    
    # Tangani kuantisasi output jika model uint8/int8
    if output_details[0]['dtype'] == np.uint8:
        scale, zero_point = output_details[0]['quantization']
        output_data = scale * (output_data.astype(np.float32) - zero_point)
    elif output_details[0]['dtype'] == np.int8:
        scale, zero_point = output_details[0]['quantization']
        output_data = scale * (output_data.astype(np.float32) - zero_point)

    # Konversi output ke rentang probabilitas 0 - 100%
    if np.max(output_data) > 1.0:
        # Jika model mengeluarkan bentuk logit/raw score yang besar
        exp_vals = np.exp(output_data - np.max(output_data))
        probabilities = exp_vals / np.sum(exp_vals)
    else:
        probabilities = output_data

    predicted_index = np.argmax(probabilities)
    confidence = float(probabilities[predicted_index]) * 100
    
    # Tampilkan Hasil Utama
    predicted_label = labels[predicted_index] if predicted_index < len(labels) else "Tidak Diketahui"
    
    st.subheader("Hasil Prediksi:")
    st.success(f"**Jenis Sampah: {predicted_label.upper()}** (Akurasi: {confidence:.2f}%)")
    
    # Debug rincian untuk melihat apakah skornya benar-benar stuck
    with st.expander("Cek Detail Probabilitas Murni"):
        for i, lab in enumerate(labels):
            st.write(f"- {lab}: {float(probabilities[i])*100:.2f}%")
