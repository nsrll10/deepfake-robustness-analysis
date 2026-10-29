import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import io

st.set_page_config(page_title="Deepfake Lab Test", page_icon="🕵️", layout="centered")
st.title(" Pengujian Deteksi Deepfake")
st.markdown("Alat uji laboratorium dengan fitur **Auto Face-Cropping** untuk menganalisis ketahanan model terhadap kompresi citra.")
st.markdown("---")


@st.cache_resource
def load_model():
    model_path = 'mobilenetv2_celebdf.keras'
    return tf.keras.models.load_model(model_path)

try:
    model = load_model()
    st.sidebar.success("✅ Model MobileNetV2 (Robust Version) berhasil dimuat!")
except Exception as e:
    st.sidebar.error(f"❌ Gagal memuat model: {e}")
    st.stop()

class_names = ['Deepfake', 'Real']


st.sidebar.header("⚙️ Skenario Pengujian")
quality_factor = st.sidebar.slider(
    "Tingkat Kompresi JPEG (Quality Factor)", 
    min_value=10, 
    max_value=100, 
    value=100, 
    step=10
)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def preprocess_and_crop(img_pil, quality):
    # Konversi PIL ke array BGR untuk OpenCV
    img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    
    # Lacak posisi wajah
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))
    
    if len(faces) == 0:
        return None, None, None, "Sistem menolak: Tidak ada wajah manusia yang terdeteksi dengan jelas."
    
    # Ambil wajah pertama dan lakukan pemotongan
    (x, y, w, h) = faces[0]
    margin = int(w * 0.2)
    
    y1, y2 = max(0, y - margin), min(img_cv.shape[0], y + h + margin)
    x1, x2 = max(0, x - margin), min(img_cv.shape[1], x + w + margin)
    
    cropped_face_cv = img_cv[y1:y2, x1:x2]
    
    # SIMPAN WAJAH ASLI (SEBELUM KOMPRESI)
    uncompressed_face_pil = Image.fromarray(cv2.cvtColor(cropped_face_cv, cv2.COLOR_BGR2RGB))
    
    # SIMULASI KOMPRESI JPEG PADA WAJAH
    buffer = io.BytesIO()
    uncompressed_face_pil.save(buffer, format="JPEG", quality=quality)
    compressed_face_pil = Image.open(buffer)
    
    # Resize dan Normalisasi untuk AI (224x224)
    img_resized = compressed_face_pil.resize((224, 224))
    img_array = np.array(img_resized) / 255.0
    img_batch = np.expand_dims(img_array, axis=0)
    
    # Kembalikan ketiga variabel: tensor model, wajah asli, wajah kompresi
    return img_batch, uncompressed_face_pil, compressed_face_pil, "Sukses"


uploaded_file = st.file_uploader("Unggah Citra / Foto Latar Penuh (JPG/PNG)", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    original_img = Image.open(uploaded_file).convert('RGB')
    
    # Menampilkan gambar utuh sebagai referensi kecil
    with st.expander("Lihat Gambar Utuh (Latar Penuh)"):
        st.image(original_img, use_container_width=True)
    
    st.markdown("---")
    
    if st.button("🔍 Ekstrak Wajah & Analisis Model", use_container_width=True):
        with st.spinner("Sistem sedang melacak wajah dan menyimulasikan kompresi..."):
            
            # Eksekusi fungsi pelacak wajah
            img_tensor, uncompressed_display, compressed_display, status_msg = preprocess_and_crop(original_img, quality_factor)
            
            if img_tensor is None:
                st.error("⚠️ PROSES DIBATALKAN")
                st.warning(status_msg)
            else:
                st.markdown("**Hasil Ekstraksi & Simulasi Kompresi Wajah**")
                
                # TAMPILAN BERSEBELAHAN (KIRI ASLI, KANAN KOMPRESI)
                col1, col2 = st.columns(2)
                with col1:
                    st.image(uncompressed_display, caption="Wajah Asli (Kualitas 100%)", use_container_width=True)
                with col2:
                    st.image(compressed_display, caption=f"Wajah Terkompresi (QF: {quality_factor})", use_container_width=True)
                
                st.markdown("---")
                
                predictions = model.predict(img_tensor)[0]
                predicted_index = np.argmax(predictions)
                predicted_class = class_names[predicted_index]
                confidence = predictions[predicted_index] * 100
                
                st.subheader("📊 Hasil Prediksi (Pada Citra Terkompresi)")
                if predicted_class == 'Real':
                    st.success(f"**KESIMPULAN: {predicted_class.upper()}** (Tingkat Keyakinan: {confidence:.2f}%)")
                else:
                    st.error(f"**KESIMPULAN: {predicted_class.upper()}** (Tingkat Keyakinan: {confidence:.2f}%)")
          
                st.write("**Detail Probabilitas:**")
                st.write(f"- Manipulasi (Deepfake): {predictions[0] * 100:.2f}%")
                st.write(f"- Asli (Real): {predictions[1] * 100:.2f}%")