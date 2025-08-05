import streamlit as st
from ultralytics import YOLO
import numpy as np
import os
from PIL import Image
import gdown
import tempfile
import cv2
from io import BytesIO

# 📥 Download model from Google Drive if not present
MODEL_PATH = "best.pt"
if not os.path.exists(MODEL_PATH):
    st.info("📦 Downloading model from Google Drive...")
    file_id = "18O0SvhxoP1hCWkRUPUKsDR1E2bCPLVMR" 
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, MODEL_PATH, quiet=False)

# 🧠 Load YOLOv8 model with caching
@st.cache_resource
def load_model(path):
    return YOLO(path)

try:
    model = load_model(MODEL_PATH)
except Exception as e:
    st.error(f"❌ Failed to load model: {e}")
    st.stop()

# 🧠 App UI
st.set_page_config(page_title="HackByte Detector", layout="centered")
st.title("🚀 HackByte Object Detector")
st.markdown("Upload an image to detect space station objects with high precision.")

# 📷 Image Upload
uploaded_file = st.file_uploader("📁 Choose an image", type=["jpg", "jpeg", "png", "webp"])

if uploaded_file:
    try:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="🖼️ Uploaded Image", use_container_width=True)

        # 💾 Save image temporarily
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            image.save(tmp.name)
            image_path = tmp.name

        # 🔍 Run detection
        with st.spinner("🔍 Detecting objects..."):
            results = model.predict(image_path)

        # 🖼️ Show annotated image(s)
        for r in results:
            annotated_img = r.plot()
            st.image(annotated_img, caption="✅ Detections", use_column_width=True)

            # 💾 Download button for annotated image
            is_success, buffer = cv2.imencode(".jpg", annotated_img)
            if is_success:
                st.download_button(
                    label="📥 Download Annotated Image",
                    data=BytesIO(buffer.tobytes()),
                    file_name="detection_result.jpg",
                    mime="image/jpeg"
                )

            # 📋 Show detection info
            with st.expander("🔍 View Detection Info"):
                for box in r.boxes.data.tolist():
                    cls_id = int(box[5])
                    conf = float(box[4])
                    label = model.names.get(cls_id, f"Class {cls_id}")
                    st.write(f"→ **{label}** ({conf:.2f})")

    except Exception as e:
        st.error(f"❌ Error processing image: {e}")
