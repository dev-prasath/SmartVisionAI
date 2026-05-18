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

from download_models import download_models

download_models()

print("Testing completed successfully!")



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

# =========================================================
# LOAD MOBILENET
# =========================================================

@st.cache_resource
def load_mobilenet_model():

    model = create_mobilenet_model()

    model.load_weights(
        "models/mobilenet_weights.weights.h5"
    )

    print("✅ MobileNet Loaded")

    return model


# =========================================================
# LOAD RESNET50
# =========================================================

@st.cache_resource
def load_resnet_model():

    model = create_resnet_model()

    model.load_weights(
        "models/resnet50_weights.weights.h5"
    )

    print("✅ ResNet50 Loaded")

    return model


# =========================================================
# LOAD EFFICIENTNET
# =========================================================

@st.cache_resource
def load_effnet_model():

    model = create_efficientnet_model()

    model.load_weights(
        "models/efficientnet_weights.weights.h5"
    )

    print("✅ EfficientNet Loaded")

    return model


# @st.cache_resource
# def load_models():

#     # =====================================================
#     # MOBILE NET
#     # =====================================================

#     mobilenet_model = create_mobilenet_model()

#     mobilenet_model.load_weights(

#         "models/mobilenet_weights.weights.h5"
#     )

#     print("✅ MobileNet Loaded")


#     # =====================================================
#     # RESNET50
#     # =====================================================

#     resnet_model = create_resnet_model()

#     resnet_model.load_weights(

#         "models/resnet50_weights.weights.h5"
#     )

#     print("✅ ResNet50 Loaded")


#     # =====================================================
#     # EFFICIENTNET
#     # =====================================================

#     efficientnet_model = create_efficientnet_model()

#     efficientnet_model.load_weights(

#         "models/efficientnet_weights.weights.h5"
#     )

#     print("✅ EfficientNet Loaded")


#     return (

#         mobilenet_model,

#         resnet_model,

#         efficientnet_model
#     )


# mobilenet_model, resnet_model, efficientnet_model = load_models()

# =========================================================
# LOAD YOLO MODEL
# =========================================================


@st.cache_resource
def load_yolo_model():

    return YOLO("models/best .pt")


# yolo_model = load_yolo_model()

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

    # =====================================================
    # MODEL SELECTION
    # =====================================================

    selected_model = st.selectbox(

        "Choose Classification Model",

        [
            "MobileNetV2",
            "ResNet50",
            "EfficientNetB0"
        ]
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
                    "Running Deep Learning Model..."
                ):

                    processed_image = preprocess_image(
                        image
                    )

                    # =====================================
                    # LOAD SELECTED MODEL
                    # =====================================

                    if selected_model == "MobileNetV2":

                        model = load_mobilenet_model()

                    elif selected_model == "ResNet50":

                        model = load_resnet_model()

                    else:

                        model = load_effnet_model()

                    # =====================================
                    # PREDICTION
                    # =====================================

                    prediction = model.predict(
                        processed_image
                    )

                    predicted_class = np.argmax(
                        prediction
                    )

                    confidence = np.max(
                        prediction
                    )

                    # =====================================
                    # DISPLAY RESULTS
                    # =====================================

                    st.success(
                        f"""
                        ✅ Prediction:
                        {CLASS_NAMES[predicted_class]}
                        """
                    )

                    st.info(
                        f"""
                        🎯 Confidence:
                        {confidence*100:.2f}%
                        """
                    )

                    # =====================================
                    # TOP 5 PREDICTIONS
                    # =====================================

                    top5_indices = np.argsort(
                        prediction[0]
                    )[-5:][::-1]

                    top5_classes = [
                        CLASS_NAMES[i]
                        for i in top5_indices
                    ]

                    top5_scores = [
                        prediction[0][i] * 100
                        for i in top5_indices
                    ]

                    results_df = pd.DataFrame({

                        "Class": top5_classes,

                        "Confidence (%)": [

                            round(score, 2)
                            for score in top5_scores
                        ]
                    })

                    st.subheader(
                        "📊 Top 5 Predictions"
                    )

                    st.dataframe(
                        results_df,
                        use_container_width=True
                    )

                    # =====================================
                    # CHART
                    # =====================================

                    chart_df = pd.DataFrame({

                        "Class": top5_classes,

                        "Confidence": top5_scores
                    })

                    st.subheader(
                        "📈 Confidence Scores"
                    )

                    st.bar_chart(
                        chart_df.set_index("Class")
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

            with st.spinner(
                "Running YOLOv8 Detection..."
            ):

                # =====================================
                # LOAD YOLO ONLY WHEN NEEDED
                # =====================================

                yolo_model = load_yolo_model()

                # =====================================
                # SAVE TEMP IMAGE
                # =====================================

                temp_file = tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=".jpg"
                )

                image.save(temp_file.name)

                # =====================================
                # RUN PREDICTION
                # =====================================

                results = yolo_model.predict(
                    source=temp_file.name,
                    conf=confidence_threshold
                )

                # =====================================
                # DRAW BOUNDING BOXES
                # =====================================

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

                # =====================================
                # EXTRACT DETECTIONS
                # =====================================

                boxes = results[0].boxes

                detection_data = {

                    "Object": [],

                    "Confidence (%)": []
                }

                if len(boxes) > 0:

                    for box in boxes:

                        cls_id = int(box.cls[0])

                        class_name = yolo_model.names[
                            cls_id
                        ]

                        confidence = float(
                            box.conf[0]
                        )

                        detection_data[
                            "Object"
                        ].append(class_name)

                        detection_data[
                            "Confidence (%)"
                        ].append(
                            round(confidence * 100, 2)
                        )

                    detection_df = pd.DataFrame(
                        detection_data
                    )

                    st.subheader(
                        "📊 Detection Results"
                    )

                    st.dataframe(
                        detection_df,
                        use_container_width=True
                    )

                    # =================================
                    # DETECTION COUNTS
                    # =================================

                    object_counts = detection_df[
                        "Object"
                    ].value_counts()

                    st.subheader(
                        "📈 Object Counts"
                    )

                    st.bar_chart(
                        object_counts
                    )

                else:

                    st.warning(
                        "No objects detected."
                    )
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

    st.warning(
        """
        Webcam detection works best in local system.
        Some cloud platforms may not support webcam access.
        """
    )

    enable_webcam = st.checkbox(
        "Enable Webcam"
    )

    if enable_webcam:

        # =====================================
        # LOAD YOLO ONLY WHEN NEEDED
        # =====================================

        yolo_model = load_yolo_model()

        FRAME_WINDOW = st.image([])

        camera = cv2.VideoCapture(0)

        stop_button = st.button(
            "Stop Webcam"
        )

        while camera.isOpened() and not stop_button:

            success, frame = camera.read()

            if not success:

                st.error(
                    "Unable to access webcam"
                )

                break

            # =================================
            # RUN YOLO DETECTION
            # =================================

            results = yolo_model.predict(
                source=frame,
                conf=0.25,
                verbose=False
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

        st.success(
            "Webcam stopped successfully"
        )
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