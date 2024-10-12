import streamlit as st
from fastai.vision.all import load_learner, PILImage
import pathlib
import numpy as np

# Set a background image
st.markdown(f"""
<style>
.stApp{{
    background-image: url(https://www.gaiaca.com/wp-content/uploads/2021/02/what-to-do-with-cannabis-trim.jpg);
    background-size: cover;
}}
</style>
""",unsafe_allow_html=True)

# Force FastAI to use WindowsPath instead of PosixPath
pathlib.PosixPath = pathlib.WindowsPath

# Load the model
model = load_learner("models/canhealth_model.pkl")

st.title('Cannabis Plant Disease Detection')

# Define class names
class_names = [
    'Aphids', 'Botrytis', 'Dehydration', 'Healthy', 'Leaf Miners',
    'Nitrogen Deficiency', 'Nutrient Burn', 'Overwatering', 'PH Fluctuation',
    'Phosphorus Deficiency', 'Potassium Deficiency', 'Powdery Mildew', 'Septoria']

# Upload image
uploaded_file = st.file_uploader("Upload your cannabis plant image and try out the demo disease detection model!", type=['png','jpg','svg'])
if uploaded_file is not None:
    img = PILImage.create(uploaded_file)

    # Get predictions
    pred_class, pred_idx, outputs = model.predict(img)

    st.image(img.to_thumb(224, 224), caption=f"Predicted: {pred_class}")
# import streamlit as st
# from fastai.vision.all import load_learner, PILImage
# import pathlib
# import numpy as np

# # Force FastAI to use WindowsPath instead of PosixPath
# pathlib.PosixPath = pathlib.WindowsPath

# # Load the model
# model = load_learner("models/canhealth_model.pkl")

# st.title('Cannabis Plant Disease Detection')

# # Define class names
# class_names = [
#     'Aphids', 'Botrytis', 'Dehydration', 'Healthy', 'Leaf Miners',
#     'Nitrogen Deficiency', 'Nutrient Burn', 'Overwatering', 'PH Fluctuation',
#     'Phosphorus Deficiency', 'Potassium Deficiency', 'Powdery Mildew', 'Septoria']

# # Upload image
# uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "png", "jpeg"])
# if uploaded_file is not None:
#     img = PILImage.create(uploaded_file)

#     # Get predictions
#     pred_class, pred_idx, outputs = model.predict(img)

#     # Ensure outputs is a valid array
#     if isinstance(outputs, np.ndarray) and len(outputs) > 0:
#         # Show image with prediction
#         st.image(img.to_thumb(224, 224), caption=f"Predicted: {pred_class}")

#         # Show the top 3 predicted labels
#         top_n = 3 
#         sorted_indices = np.argsort(outputs)[-1:-top_n-1:-1]

#         st.write(f"Top 3 possible diagnoses:")
#         for i in sorted_indices:
#             st.write(f"{class_names[i]}: {outputs[i]:.2f} probability")
#     else:
#         st.error("The model did not return valid predictions.")
