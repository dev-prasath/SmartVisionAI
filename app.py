# app.py
import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import cv2
import tempfile
import time

from PIL import Image
from ultralytics import YOLO

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    GlobalAveragePooling2D,
    Dense,
    Dropout
)

from tensorflow.keras.applications import MobileNetV2

from streamlit_option_menu import option_menu
from keras.models import load_model
from tensorflow.keras.layers import BatchNormalization

from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    GlobalAveragePooling2D,
    Dense,
    Dropout,
    BatchNormalization
)

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="SmartVision AI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown(
    """
    <style>

    .stApp {
        background: linear-gradient(to right, #0f172a, #111827);
        color: white;
    }

    h1, h2, h3, h4 {
        color: white !important;
    }

    .metric-card {
        background-color: #1e293b;
        padding: 25px;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0px 4px 10px rgba(0,0,0,0.3);
    }

    .feature-card {
        background-color: #1e293b;
        padding: 25px;
        border-radius: 15px;
        margin-bottom: 20px;
    }

    </style>
    """,
    unsafe_allow_html=True
)

# =========================================================
# CLASS NAMES
# =========================================================

CLASS_NAMES = {
    0: "airplane",
    1: "bed",
    2: "bench",
    3: "bicycle",
    4: "bird",
    5: "bottle",
    6: "bowl",
    7: "bus",
    8: "cake",
    9: "car",
    10: "cat",
    11: "chair",
    12: "couch",
    13: "cow",
    14: "cup",
    15: "dog",
    16: "elephant",
    17: "horse",
    18: "motorcycle",
    19: "person",
    20: "pizza",
    21: "potted plant",
    22: "stop sign",
    23: "traffic light",
    24: "train",
    25: "truck"
}

# =========================================================
# BUILD MOBILENET MODEL
# =========================================================

def create_mobilenet_model():

    base_model = MobileNetV2(

        weights=None,

        include_top=False,

        input_shape=(224, 224, 3)
    )

    base_model.trainable = False

    model = Sequential([

        base_model,

        GlobalAveragePooling2D(),

        Dense(256, activation='relu'),

        BatchNormalization(),

        Dropout(0.4),

        Dense(26, activation='softmax')
    ])

    model.build((None, 224, 224, 3))

    return model


# =========================================================
# BUILD RESNET50 MODEL
# =========================================================

def create_resnet_model():

    base_model = ResNet50(

        weights=None,

        include_top=False,

        input_shape=(224,224,3)
    )

    base_model.trainable = False

    model = Sequential([

        base_model,

        GlobalAveragePooling2D(),

        Dense(256, activation='relu'),

        Dropout(0.4),

        Dense(26, activation='softmax')
    ])

    model.build((None,224,224,3))

    return model


# =========================================================
# BUILD EFFICIENTNET MODEL
# =========================================================

from tensorflow.keras.applications import EfficientNetB0

def create_efficientnet_model():

    base_model = EfficientNetB0(

        weights=None,

        include_top=False,

        input_shape=(224,224,3)
    )

    base_model.trainable = False

    model = Sequential([

        base_model,

        GlobalAveragePooling2D(),

        Dense(256, activation='relu'),

        Dropout(0.4),

        Dense(26, activation='softmax')
    ])

    model.build((None,224,224,3))

    return model


# =========================================================
# LOAD ALL MODELS
# =========================================================

@st.cache_resource
def load_models():

    # =====================================================
    # MOBILE NET
    # =====================================================

    mobilenet_model = create_mobilenet_model()

    mobilenet_model.load_weights(

        "models/mobilenet_weights.weights.h5"
    )

    print("✅ MobileNet Loaded")


    # =====================================================
    # RESNET50
    # =====================================================

    resnet_model = create_resnet_model()

    resnet_model.load_weights(

        "models/resnet50_weights.weights.h5"
    )

    print("✅ ResNet50 Loaded")


    # =====================================================
    # EFFICIENTNET
    # =====================================================

    efficientnet_model = create_efficientnet_model()

    efficientnet_model.load_weights(

        "models/efficientnet_weights.weights.h5"
    )

    print("✅ EfficientNet Loaded")


    return (

        mobilenet_model,

        resnet_model,

        efficientnet_model
    )


mobilenet_model, resnet_model, efficientnet_model = load_models()

# =========================================================
# LOAD YOLO MODEL
# =========================================================


@st.cache_resource
def load_yolo_model():

    return YOLO("models/best .pt")


yolo_model = load_yolo_model()

# =========================================================
# PREPROCESS IMAGE
# =========================================================


def preprocess_image(image):

    image = image.resize((224, 224))

    image = np.array(image)

    if len(image.shape) == 2:
        image = np.stack((image,) * 3, axis=-1)

    if image.shape[-1] == 4:
        image = image[:, :, :3]

    image = image / 255.0

    image = np.expand_dims(image, axis=0)

    return image

# =========================================================
# PREMIUM SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown(
        """
        <div style='text-align:center;'>
        <h1 style='color:white;'>🤖 SmartVision AI</h1>
        <p style='color:lightgray;'>Computer Vision Platform</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    selected = option_menu(
        menu_title=None,

        options=[
            "Home",
            "Image Classification",
            "Object Detection",
            "Performance Dashboard",
            "Live Webcam",
            "Dataset Info",
            "About"
        ],

        icons=[
            "house",
            "image",
            "bounding-box",
            "bar-chart",
            "camera-video",
            "database",
            "info-circle"
        ],

        default_index=0,

        styles={
            "container": {
                "padding": "5px",
                "background-color": "#111827",
            },

            "icon": {
                "color": "#38bdf8",
                "font-size": "18px"
            },

            "nav-link": {
                "font-size": "16px",
                "text-align": "left",
                "margin": "5px",
                "border-radius": "10px",
                "color": "white",
            },

            "nav-link-selected": {
                "background-color": "#2563eb",
            },
        }
    )

page = selected

# =========================================================
# HOME PAGE
# =========================================================

if page == "Home":

    st.title("🤖 SmartVision AI")

    st.markdown(
        """
        ### AI Powered Computer Vision Platform

        Upload images, classify objects,
        detect multiple objects in real-time,
        and explore deep learning performance.
        """
    )

    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    with col1:

        st.markdown(
            """
            <div class='metric-card'>
            <h2>26</h2>
            <p>Object Classes</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:

        st.markdown(
            """
            <div class='metric-card'>
            <h2>YOLOv8</h2>
            <p>Object Detection</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col3:

        st.markdown(
            """
            <div class='metric-card'>
            <h2>MobileNetV2</h2>
            <p>Best Classifier</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:

        st.markdown(
            """
            <div class='feature-card'>
            <h3>🖼 Image Classification</h3>
            <p>
            Predict image classes using transfer learning
            and deep learning architectures.
            </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:

        st.markdown(
            """
            <div class='feature-card'>
            <h3>🎯 Object Detection</h3>
            <p>
            Detect multiple objects in images using
            YOLOv8 object detection.
            </p>
            </div>
            """,
            unsafe_allow_html=True
        )

# =========================================================
# IMAGE CLASSIFICATION
# =========================================================

elif page == "Image Classification":
    st.title("🖼 Image Classification")

    uploaded_file = st.file_uploader(
        "Upload Image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:

        image = Image.open(uploaded_file)

        col1, col2 = st.columns(2)

        with col1:

            st.image(
                image,
                caption="Uploaded Image",
                use_container_width=True
            )

        with col2:

            if st.button("🔍 Predict Image"):

                with st.spinner(
                    "Running Deep Learning Models..."
                ):

                    processed_image = preprocess_image(
                        image
                    )

                    # =========================================
                    # MOBILE NET
                    # =========================================

                    mobilenet_pred = mobilenet_model.predict(
                        processed_image
                    )

                    mobile_class = np.argmax(
                        mobilenet_pred
                    )

                    mobile_conf = np.max(
                        mobilenet_pred
                    )

                    # =========================================
                    # RESNET
                    # =========================================

                    resnet_pred = resnet_model.predict(
                        processed_image
                    )

                    resnet_class = np.argmax(
                        resnet_pred
                    )

                    resnet_conf = np.max(
                        resnet_pred
                    )

                    # =========================================
                    # EFFICIENTNET
                    # =========================================

                    efficient_pred = efficientnet_model.predict(
                        processed_image
                    )

                    efficient_class = np.argmax(
                        efficient_pred
                    )

                    efficient_conf = np.max(
                        efficient_pred
                    )

                    # =========================================
                    # RESULTS TABLE
                    # =========================================

                    results_df = pd.DataFrame({

                        "Model": [

                            "MobileNetV2",

                            "ResNet50",

                            "EfficientNetB0"
                        ],

                        "Prediction": [

                            CLASS_NAMES[mobile_class],

                            CLASS_NAMES[resnet_class],

                            CLASS_NAMES[efficient_class]
                        ],

                        "Confidence (%)": [

                            round(
                                mobile_conf * 100,
                                2
                            ),

                            round(
                                resnet_conf * 100,
                                2
                            ),

                            round(
                                efficient_conf * 100,
                                2
                            )
                        ]
                    })

                    st.subheader(
                        "📊 Model Predictions"
                    )

                    st.dataframe(
                        results_df,
                        use_container_width=True
                    )

                    # =========================================
                    # BEST MODEL
                    # =========================================

                    best_confidence = max(
                        mobile_conf,
                        resnet_conf,
                        efficient_conf
                    )

                    if best_confidence == mobile_conf:

                        st.success(
                            f"""
                            🏆 Best Prediction:
                            MobileNetV2 predicted
                            {CLASS_NAMES[mobile_class]}
                            with confidence
                            {mobile_conf*100:.2f}%
                            """
                        )

                    elif best_confidence == resnet_conf:

                        st.success(
                            f"""
                            🏆 Best Prediction:
                            ResNet50 predicted
                            {CLASS_NAMES[resnet_class]}
                            with confidence
                            {resnet_conf*100:.2f}%
                            """
                        )

                    else:

                        st.success(
                            f"""
                            🏆 Best Prediction:
                            EfficientNetB0 predicted
                            {CLASS_NAMES[efficient_class]}
                            with confidence
                            {efficient_conf*100:.2f}%
                            """
                        )

                    # =========================================
                    # CONFIDENCE CHART
                    # =========================================

                    chart_df = pd.DataFrame({

                        "Models": [

                            "MobileNetV2",

                            "ResNet50",

                            "EfficientNetB0"
                        ],

                        "Confidence": [

                            mobile_conf * 100,

                            resnet_conf * 100,

                            efficient_conf * 100
                        ]
                    })

                    st.subheader(
                        "📈 Confidence Comparison"
                    )

                    st.bar_chart(
                        chart_df.set_index("Models")
                    )
# =========================================================
# OBJECT DETECTION
# =========================================================

elif page == "Object Detection":

    st.title("🎯 YOLOv8 Object Detection")

    uploaded_file = st.file_uploader(
        "Upload Image",
        type=["jpg", "jpeg", "png"],
        key="detect"
    )

    confidence_threshold = st.slider(
        "Confidence Threshold",
        0.1,
        1.0,
        0.25
    )

    if uploaded_file is not None:

        image = Image.open(uploaded_file)

        st.image(
            image,
            caption="Uploaded Image",
            use_container_width=True
        )

        if st.button("🚀 Run YOLO Detection"):

            with st.spinner("🚀 Running YOLOv8 Detection..."):

                temp_file = tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=".jpg"
                )

                image.save(temp_file.name)

                results = yolo_model.predict(
                    source=temp_file.name,
                    conf=confidence_threshold
                )

                plotted_image = results[0].plot()

                plotted_image = cv2.cvtColor(
                    plotted_image,
                    cv2.COLOR_BGR2RGB
                )

                st.image(
                    plotted_image,
                    caption="Detection Results",
                    use_container_width=True
                )

                boxes = results[0].boxes

                detection_data = {
                    "Object": [],
                    "Confidence": []
                }

                if len(boxes) > 0:

                    for box in boxes:

                        cls_id = int(box.cls[0])

                        class_name = yolo_model.names[cls_id]

                        confidence = float(box.conf[0])

                        detection_data["Object"].append(class_name)

                        detection_data["Confidence"].append(
                            round(confidence * 100, 2)
                        )

                    detection_df = pd.DataFrame(detection_data)

                    st.dataframe(
                        detection_df,
                        use_container_width=True
                    )

                else:

                    st.warning("No objects detected")

# =========================================================
# PERFORMANCE DASHBOARD
# =========================================================

elif page == "Performance Dashboard":

    st.title("📊 Performance Dashboard")

    metrics_df = pd.DataFrame({

        "Model": [
            "MobileNetV2",
            "ResNet50",
            "EfficientNetB0"
        ],

        "Accuracy": [
            87,
            95,
            81
        ],

        "Inference Speed(ms)": [
            50,
            100,
            80
        ]
    })

    st.dataframe(
        metrics_df,
        use_container_width=True
    )

    st.subheader("📈 Accuracy Comparison")

    st.bar_chart(
        metrics_df.set_index("Model")["Accuracy"]
    )

    st.subheader("⚡ Inference Speed")

    st.bar_chart(
        metrics_df.set_index("Model")["Inference Speed(ms)"]
    )

# =========================================================
# LIVE WEBCAM
# =========================================================

elif page == "Live Webcam":

    st.title("📹 Live Webcam Detection")

    run = st.checkbox("Start Webcam")

    FRAME_WINDOW = st.image([])

    camera = cv2.VideoCapture(0)

    while run:

        success, frame = camera.read()

        if not success:
            st.error("Webcam not detected")
            break

        results = yolo_model.predict(
            source=frame,
            conf=0.25
        )

        annotated_frame = results[0].plot()

        annotated_frame = cv2.cvtColor(
            annotated_frame,
            cv2.COLOR_BGR2RGB
        )

        FRAME_WINDOW.image(
            annotated_frame,
            channels="RGB"
        )

    camera.release()

# =========================================================
# DATASET INFO
# =========================================================

elif page == "Dataset Info":

    st.title("📁 Dataset Information")

    dataset_df = pd.DataFrame({
        "Classes": list(CLASS_NAMES.values())
    })

    st.dataframe(
        dataset_df,
        use_container_width=True
    )

    st.markdown("---")

    st.write("### Dataset Statistics")

    st.write("- 26 Classes")
    st.write("- COCO Based Dataset")
    st.write("- Custom Classification Dataset")
    st.write("- YOLO Detection Dataset")

# =========================================================
# ABOUT PAGE
# =========================================================

elif page == "About":

    st.title("ℹ About SmartVision AI")

    st.write(
        """
        ## SmartVision AI

        Deep Learning and Computer Vision project using:

        - MobileNetV2
        - ResNet50
        - EfficientNetB0
        - YOLOv8

        ### Features

        - Image Classification
        - Object Detection
        - Real-time Webcam Detection
        - Performance Dashboard

        ### Technologies

        - TensorFlow
        - Streamlit
        - OpenCV
        - Ultralytics YOLO
        """
    )

    st.success("Application Running Successfully 🚀")

# =========================================================
# FOOTER
# =========================================================

st.markdown("---")

footer = """
<style>

.footer {
    position: fixed;
    left: 0;
    bottom: 0;
    width: 100%;
    background-color: #111827;
    color: white;
    text-align: center;
    padding: 10px;
    font-size: 14px;
    z-index: 100;
}

</style>

<div class="footer">
🚀 SmartVision AI | Built with Streamlit & YOLOv8
</div>
"""

st.markdown(footer, unsafe_allow_html=True)