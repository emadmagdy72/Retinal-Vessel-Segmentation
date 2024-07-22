import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

st.set_page_config(page_title="Retina Image Segmentation", layout="wide")

st.markdown(
    """
    <style>
    .main {
        background-color: #f5f5f5;
        font-family: Arial, sans-serif;
    }
    .title {
        text-align: center;
        font-size: 2.5rem;
        color: #4CAF50;
        margin-top: 20px;
        margin-bottom: 20px;
    }
    .sidebar .sidebar-content {
        background-color: #f5f5f5;
        color: #333;
    }
    .sidebar .sidebar-content h1 {
        color: #4CAF50;
    }
    .button {
        background-color: #4CAF50;
        color: white;
        border: none;
        padding: 10px 20px;
        text-align: center;
        font-size: 1rem;
        margin: 10px 2px;
        cursor: pointer;
        border-radius: 5px;
    }
    .button:hover {
        background-color: #45a049;
    }
    .performance-text {
        color: black;
    }
    .performance-table {
        width: 100%;
        border-collapse: collapse;
        color: black;
    }
    .performance-table, .performance-table th, .performance-table td {
        border: 1px solid black;
    }
    .performance-table th, .performance-table td {
        padding: 10px;
        text-align: left;
    }
    .performance-table th {
        background-color: #4CAF50;
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

@st.cache_resource
def load_model(path):
    return tf.keras.models.load_model(path)

unet_model = load_model("./models/unet_model.h5")
sagenet_model = load_model("./models/sgnet_model.h5")

def preprocess_retina_image(image):
    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image = cv2.resize(image, (256, 256))
    image = image / 255.0
    return np.expand_dims(image, axis=0)

def preprocess_mask_image(mask):
    mask = cv2.resize(mask, (256, 256))
    mask = mask / 255.0
    return np.expand_dims(mask, axis=-1)

sample_images = {
    "Sample 1": "./sample_images/1.png",
    "Sample 2": "./sample_images/10.png",
    "Sample 3": "./sample_images/1003.png",
}


performance_metrics = {
    "U-Net": {
        "Overall": {"Accuracy": 0.97},
        "Class-wise": {
            "Class Background": {"Precision": 0.97, "Recall": 1, "F1-Score": 0.98},
            "Class Objetc": {"Precision": 0.95, "Recall": 0.67, "F1-Score": 0.78},
        }
    },
    "Sagenet": {
         "Overall": {"Accuracy": 0.98},
        "Class-wise": {
            "Class Background": {"Precision": 0.98, "Recall": 0.99, "F1-Score": 0.99},
            "Class Objetc": {"Precision": 0.92, "Recall": 0.82, "F1-Score": 0.87},
        }     

    }
}

st.sidebar.header("Navigation")
page = st.sidebar.selectbox("Select a page", ["Segmentation", "Performance Comparison"])

if page == "Segmentation":
    st.markdown('<div class="title">Retina Image Segmentation</div>', unsafe_allow_html=True)

    st.sidebar.header("Upload or Select Sample Image")
    image_source = st.sidebar.radio("Choose Image Source", ("Upload", "Sample"))
    if image_source == "Upload":
        uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    else:
        sample_choice = st.sidebar.selectbox("Choose a sample image", list(sample_images.keys()))

    model_choice = st.sidebar.selectbox("Choose the model", ("U-Net", "Sagenet"))

    if st.sidebar.button("Segment Image"):
        if image_source == "Upload" and uploaded_file is not None:
            image = Image.open(uploaded_file)
        elif image_source == "Sample" and sample_choice is not None:
            image_path = sample_images[sample_choice]
            image = Image.open(image_path)

        if image is not None:
            input_image = preprocess_retina_image(image)

            if model_choice == "U-Net":
                model = unet_model
            else:
                model = sagenet_model

            with st.spinner('Segmenting...'):
                prediction = model.predict(input_image)

            segmented_image = (prediction[0] * 255).astype(np.uint8)
            segmented_image = cv2.cvtColor(segmented_image, cv2.COLOR_GRAY2RGB)
            segmented_pil_image = Image.fromarray(segmented_image)

            col1, col2 = st.columns(2)
            col1.image(image, caption='Original Image', use_column_width=True)
            col2.image(segmented_pil_image, caption='Segmented Image', use_column_width=True)
        else:
            st.warning("Please upload an image or select a sample to start segmentation.")
    else:
        st.write("Click 'Segment Image' to see the results.")

elif page == "Performance Comparison":
    st.markdown('<div class="title">Model Performance Comparison</div>', unsafe_allow_html=True)
    
    st.markdown('<h2 class="performance-text">Overall Performance Metrics</h2>', unsafe_allow_html=True)
    overall_metrics = performance_metrics["U-Net"]["Overall"].keys()
    overall_data = {
        "Metric": list(overall_metrics),
        "U-Net": [performance_metrics["U-Net"]["Overall"][metric] for metric in overall_metrics],
        "Sagenet": [performance_metrics["Sagenet"]["Overall"][metric] for metric in overall_metrics]
    }
    st.markdown('<table class="performance-table"><thead><tr><th>Metric</th><th>U-Net</th><th>Sagenet</th></tr></thead><tbody>' +
                ''.join([f'<tr><td>{metric}</td><td>{unet}</td><td>{sagenet}</td></tr>'
                         for metric, unet, sagenet in zip(overall_data['Metric'], overall_data['U-Net'], overall_data['Sagenet'])]) +
                '</tbody></table>', unsafe_allow_html=True)

    st.markdown('<h2 class="performance-text">Class-wise Performance Metrics</h2>', unsafe_allow_html=True)
    for class_name in performance_metrics["U-Net"]["Class-wise"].keys():
        st.markdown(f'<h3 class="performance-text">{class_name}</h3>', unsafe_allow_html=True)
        class_metrics = performance_metrics["U-Net"]["Class-wise"][class_name].keys()
        class_data = {
            "Metric": list(class_metrics),
            "U-Net": [performance_metrics["U-Net"]["Class-wise"][class_name][metric] for metric in class_metrics],
            "Sagenet": [performance_metrics["Sagenet"]["Class-wise"][class_name][metric] for metric in class_metrics]
        }
        st.markdown('<table class="performance-table"><thead><tr><th>Metric</th><th>U-Net</th><th>Sagenet</th></tr></thead><tbody>' +
                    ''.join([f'<tr><td>{metric}</td><td>{unet}</td><td>{sagenet}</td></tr>'
                             for metric, unet, sagenet in zip(class_data['Metric'], class_data['U-Net'], class_data['Sagenet'])]) +
                    '</tbody></table>', unsafe_allow_html=True)
