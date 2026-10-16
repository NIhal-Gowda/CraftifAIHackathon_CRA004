import cv2
import numpy as np
import streamlit as st
from ultralytics import YOLO

# Page configuration
st.set_page_config(page_title="Retail Shelf & Queue Monitor", layout="wide")

# Load YOLOv8 model
@st.cache_resource
def load_model():
    return YOLO('yolov8n.pt')

model = load_model()

st.title("🛒 Retail Shelf & Queue Monitor (PipeGen Track)")
st.markdown("Automated retail analytics for detecting queue lengths and shelf stock status using computer vision.")

# Sidebar controls
st.sidebar.header("Configuration Panel")
queue_threshold = st.sidebar.slider("Queue Alert Threshold (People)", min_value=1, max_value=10, value=3)

# File uploader for test image
uploaded_file = st.sidebar.file_uploader("Upload Store/Queue Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read the image file
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Run YOLO detection
    results = model(image_rgb)
    
    person_count = 0
    annotated_image = image_rgb.copy()
    
    for r in results:
        boxes = r.boxes
        for box in boxes:
            cls = int(box.cls[0])
            if cls == 0:  # Class 0 is 'person'
                person_count += 1
        
        # Get annotated image with bounding boxes drawn by YOLO
        annotated_image = r.plot()

    # Layout metrics
    col1, col2 = st.columns(2)
    with col1:
        st.metric(label="Detected Queue Length", value=f"{person_count} People")
    with col2:
        if person_count > queue_threshold:
            st.error(f"⚠️ ALERT: Long Queue Detected! ({person_count} > {queue_threshold})")
        else:
            st.success("✅ Queue Status: Normal")

    # Display image
    st.image(annotated_image, caption="Analyzed Retail Stream Frame", use_column_width=True)

else:
    st.info("👈 Please upload a sample retail/queue image from the sidebar to test the pipeline.")
    st.markdown("### How to test:")
    st.markdown("1. Download a picture of a checkout line or grocery store from Google.")
    st.markdown("2. Upload it using the sidebar option on the left.")
    st.markdown("3. View live object detection bounding boxes and queue alerts instantly!")
    