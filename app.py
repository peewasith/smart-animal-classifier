import streamlit as st
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing import image
from keras.applications import vgg16
import numpy as np

# ==============================
# ตั้งค่า Streamlit
# ==============================
st.set_page_config(
    page_title="ระบบจำแนกสัตว์อัจฉริยะ",
    page_icon="🐾",
    layout="wide"
)

# ==============================
# โหลดโมเดลจาก structure + weights
# ==============================
with open("model/model_structure.json", "r") as f:
    model_structure = f.read()

model = model_from_json(model_structure)
model.load_weights("model/model.weights.h5")

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
uploaded_files = st.file_uploader("เลือกภาพของคุณ (สามารถเลือกได้หลายรูป)", type=["jpg","jpeg","png"], accept_multiple_files=True)

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
            confidence = results[0][predicted_class]*100
            
            st.markdown(f"<h3 style='text-align: center; color: #4B0082;'>{predicted_name}</h3>", unsafe_allow_html=True)
            st.progress(int(confidence))  # แก้ float → int
            st.caption(f"ความมั่นใจ: {confidence:.2f}%")