import streamlit as st
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing import image
from keras.applications import vgg16
import numpy as np
import os
import gdown

# ==============================
# ตั้งค่า Streamlit
# ==============================
st.set_page_config(
    page_title="ระบบจำแนกสัตว์อัจฉริยะ",
    page_icon="🐾",
    layout="wide"
)

# ==============================
# สร้างโฟลเดอร์ model ถ้ายังไม่มี
# ==============================
os.makedirs("model", exist_ok=True)

# ==============================
# ดาวน์โหลดโมเดลจาก Google Drive หากยังไม่มี
# ==============================
model_json_path = "model/model_structure.json"
model_weights_path = "model/model.weights.h5"

# ใช้ลิงก์แบบ uc?id= สำหรับ gdown
JSON_FILE_ID = "1zPDPfB0LKsXZHoog4m_9BRUWVr7HFkHI"
H5_FILE_ID   = "12Xc4HEky4W68UxyZVDvsBGAkyeR_OyHK"

if not os.path.exists(model_json_path) or not os.path.exists(model_weights_path):
    st.info("กำลังดาวน์โหลดโมเดล กรุณารอสักครู่...")
    gdown.download(f"https://drive.google.com/uc?id={JSON_FILE_ID}", model_json_path, quiet=False)
    gdown.download(f"https://drive.google.com/uc?id={H5_FILE_ID}", model_weights_path, quiet=False)
    st.success("ดาวน์โหลดโมเดลเรียบร้อย!")

# ==============================
# โหลดโมเดลจาก structure + weights
# ==============================
with open(model_json_path, "r") as f:
    model_structure = f.read()

model = model_from_json(model_structure)
model.load_weights(model_weights_path)

# โมเดล VGG16 สำหรับ extract features
feature_model = vgg16.VGG16(weights='imagenet', include_top=False, input_shape=(224,224,3))

# ==============================
# ชื่อ class
# ==============================
classes = ["นก", "แมว", "สุนัข"]

# ==============================
# Header
# ==============================
st.markdown("<h1 style='text-align: center; color: #4B0082;'>🐾 ระบบจำแนกสัตว์อัจฉริยะ</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: gray;'>อัปโหลดรูปนก แมว หรือสุนัข ระบบจะทำนายให้ทันที!</p>", unsafe_allow_html=True)
st.markdown("---")

# ==============================
# Upload หลายรูปพร้อมกัน
# ==============================
uploaded_files = st.file_uploader(
    "เลือกภาพของคุณ (สามารถเลือกได้หลายรูป)", 
    type=["jpg","jpeg","png"], 
    accept_multiple_files=True
)

if uploaded_files:
    cols = st.columns(len(uploaded_files))
    for i, uploaded_file in enumerate(uploaded_files):
        with cols[i]:
            st.image(uploaded_file, caption="รูปที่อัปโหลด", use_column_width=True)

            # preprocess image
            img = image.load_img(uploaded_file, target_size=(224,224))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = vgg16.preprocess_input(img_array)

            # extract features + predict
            features = feature_model.predict(img_array)
            results = model.predict(features)

            predicted_class = np.argmax(results)
            predicted_name = classes[predicted_class]
            confidence = float(results[0][predicted_class])  # convert to float

            st.markdown(f"<h3 style='text-align: center; color: #4B0082;'>{predicted_name}</h3>", unsafe_allow_html=True)
            st.progress(confidence)  # st.progress รองรับ float 0-1
            st.caption(f"ความมั่นใจ: {confidence*100:.2f}%")
