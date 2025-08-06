import streamlit as st
from ultralytics import YOLO
import numpy as np
import os
from PIL import Image
import gdown
import tempfile
import cv2
from io import BytesIO
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av

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

# 🌐 UI Configuration
st.set_page_config(page_title="HackByte Detector", layout="centered")
st.title("🚀 HackByte Object Detector")
st.markdown("Upload an image or use your webcam to detect space station objects.")

# 🎛️ Input mode toggle
mode = st.radio("Select input mode", ["📁 Upload Image", "🎥 Use Webcam"])

# ================================
# 📁 IMAGE UPLOAD SECTION
# ================================
if mode == "📁 Upload Image":
    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png", "webp"])
    if uploaded_file:
        try:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="🖼️ Uploaded Image", use_container_width=True)

            # Save temporarily
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                image.save(tmp.name)
                image_path = tmp.name

            # Run detection
            with st.spinner("🔍 Detecting objects..."):
                results = model.predict(image_path)

            # Display results
            for r in results:
                annotated_img = r.plot()
                st.image(annotated_img, caption="✅ Detections", use_column_width=True)

                # Download button
                is_success, buffer = cv2.imencode(".jpg", annotated_img)
                if is_success:
                    st.download_button(
                        label="📥 Download Annotated Image",
                        data=BytesIO(buffer.tobytes()),
                        file_name="detection_result.jpg",
                        mime="image/jpeg"
                    )

                # Detection info
                with st.expander("🔍 View Detection Info"):
                    for box in r.boxes.data.tolist():
                        cls_id = int(box[5])
                        conf = float(box[4])
                        label = model.names.get(cls_id, f"Class {cls_id}")
                        st.write(f"→ **{label}** ({conf:.2f})")

        except Exception as e:
            st.error(f"❌ Error processing image: {e}")

# ================================
# 🎥 WEBCAM DETECTION SECTION
# ================================
elif mode == "🎥 Use Webcam":
    st.info("🔴 Allow webcam access to start real-time detection.")

    class YOLOWebcamDetector(VideoProcessorBase):
        def __init__(self):
            self.model = model

        def recv(self, frame):
            img = frame.to_ndarray(format="bgr24")
            results = self.model.predict(img, verbose=False)
            annotated_img = results[0].plot()
            return av.VideoFrame.from_ndarray(annotated_img, format="bgr24")

    webrtc_streamer(
        key="yolo-webcam",
        video_processor_factory=YOLOWebcamDetector,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={"video": True, "audio": False}
    )
