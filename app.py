import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# ==============================
# PAGE SETTINGS
# ==============================

st.set_page_config(
    page_title="Digital Botanist",
    page_icon="🌿",
    layout="centered"
)

# ==============================
# TITLE
# ==============================

st.title("🌿 Digital Botanist")
st.subheader("AI-Powered Tomato Leaf Disease Detection")

st.write(
    "Upload a tomato leaf image and Digital Botanist will "
    "analyze it using a trained MobileNetV2 deep-learning model."
)

# ==============================
# CLASS NAMES
# ==============================

class_names = [
    "Tomato_Early_blight",
    "Tomato_Late_blight",
    "Tomato_Leaf_Mold",
    "Tomato_healthy"
]

# ==============================
# DISEASE INFORMATION
# ==============================

disease_info = {
    "Tomato_Early_blight": {
        "name": "Tomato Early Blight",
        "description": (
            "Early blight is a fungal disease that commonly affects "
            "tomato leaves and can reduce plant health and productivity."
        )
    },

    "Tomato_Late_blight": {
        "name": "Tomato Late Blight",
        "description": (
            "Late blight is a serious tomato disease that can spread "
            "rapidly under favorable environmental conditions."
        )
    },

    "Tomato_Leaf_Mold": {
        "name": "Tomato Leaf Mold",
        "description": (
            "Tomato leaf mold is a fungal disease that mainly affects "
            "leaves and is favored by humid conditions."
        )
    },

    "Tomato_healthy": {
        "name": "Healthy Tomato Leaf",
        "description": (
            "The AI identifies this image as a healthy tomato leaf "
            "with no detected signs of the four diseases it was trained to recognize."
        )
    }
}

# ==============================
# LOAD TRAINED MODEL
# ==============================

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        "digital_botanist_tomato_model.keras"
    )

try:
    model = load_model()

except Exception as e:
    st.error("⚠️ AI model could not be loaded.")
    st.info(
        "Make sure digital_botanist_tomato_model.keras "
        "is uploaded to the same GitHub repository as app.py."
    )
    st.stop()

# ==============================
# IMAGE UPLOAD
# ==============================

uploaded_file = st.file_uploader(
    "📷 Upload a tomato leaf image",
    type=["jpg", "jpeg", "png"]
)

# ==============================
# ANALYSIS
# ==============================

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Tomato Leaf",
        use_container_width=True
    )

    if st.button("🔍 Analyze Leaf", type="primary"):

        with st.spinner("🤖 Digital Botanist is analyzing..."):

            # Resize to model input size
            image_resized = image.resize((224, 224))

            # Convert image to NumPy
            image_array = np.array(image_resized)

            # Add batch dimension
            image_array = np.expand_dims(
                image_array,
                axis=0
            )

            # AI prediction
            predictions = model.predict(
                image_array,
                verbose=0
            )[0]

            # Highest probability
            predicted_index = np.argmax(predictions)

            predicted_class = class_names[predicted_index]

            confidence = (
                predictions[predicted_index] * 100
            )

        # ==============================
        # RESULT
        # ==============================

        st.success(
            f"🌿 Prediction: "
            f"{disease_info[predicted_class]['name']}"
        )

        st.metric(
            "AI Confidence",
            f"{confidence:.2f}%"
        )

        st.write(
            disease_info[predicted_class]["description"]
        )

        # ==============================
        # PROBABILITIES
        # ==============================

        st.subheader("📊 Prediction Probabilities")

        for i, class_name in enumerate(class_names):

            probability = predictions[i] * 100

            display_name = disease_info[
                class_name
            ]["name"]

            st.write(
                f"**{display_name}: "
                f"{probability:.2f}%**"
            )

            st.progress(
                float(predictions[i])
            )

        # ==============================
        # RESEARCH INFORMATION
        # ==============================

        st.divider()

        st.subheader("🔬 Research Model")

        st.write(
            "Digital Botanist uses transfer learning with "
            "MobileNetV2 to classify four tomato-leaf categories."
        )

        st.write(
            "**Test accuracy: 94.02%**"
        )

        st.write(
            "The accuracy was measured using 819 previously "
            "unseen test images."
        )

        st.caption(
            "Research prototype — predictions should not be "
            "treated as a substitute for expert plant diagnosis."
        )
