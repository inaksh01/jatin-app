import streamlit as st
from PIL import Image
from ultralytics import YOLO
import tempfile

st.set_page_config(page_title="HackByte Detector", layout="centered")
st.title("🚀 HackByte Object Detector")

model = YOLO("best.pt")  # Assumes best.pt is in same folder

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Input Image", use_column_width=True)

    # Save image temporarily
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        image.save(tmp.name)
        image_path = tmp.name

    # Run detection
    results = model.predict(image_path)
    for r in results:
        st.image(r.plot(), caption="Detections", use_column_width=True)

    with st.expander("🔍 View Detection Info"):
        for box in r.boxes.data.tolist():
            cls_id = int(box[5])
            conf = float(box[4])
            label = model.names[cls_id]
            st.write(f"→ {label} ({conf:.2f})")
