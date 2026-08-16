import streamlit as st

st.set_page_config(
    page_title="Digital Botanist",
    page_icon="🌿",
    layout="wide"
)

st.title("🌿 Digital Botanist")
st.subheader("AI-Based Plant Disease Detection")

st.write(
    "Upload an image of a plant leaf to analyze its health "
    "and identify possible diseases."
)

st.divider()

st.header("🔬 Leaf Analysis")

uploaded_file = st.file_uploader(
    "Upload a leaf image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    st.image(
        uploaded_file,
        caption="Uploaded Leaf",
        width=400
    )

    if st.button("🔍 Analyze Leaf"):
        st.info("AI analysis will be added soon.")

st.divider()

st.caption(
    "Digital Botanist — Biology + Artificial Intelligence Research Project"
)
