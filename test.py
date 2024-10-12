# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import seaborn as sns
import plotly.express as px
import pickle

st.title('Welcome to the cannabis plant disease detection app!')
st.write("*Coming soon..*")

with open("models/canhealth_model.pkl", "rb") as f:
    model = pickle.load(f)

st.title('Try the demo detecter for your cannabis plant!')

# Upload image as input
img_variable = st.file_uploader('Upload your plant picture',type=['png','jpg','svg'])
from PIL import Image
if img_variable is not None:
    st.image(Image.open(img_variable))

if st.button("Predict"):
    st.write(f'My model predict that your cannabis plant is/has {model.predict(testing_preprocess)[0]}')
else:
    st.write("Click on Predict to view the diagnosis")

