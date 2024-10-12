from fastai.vision.all import load_learner, PILImage
from keras_preprocessing.image import load_img

# Load the model
model = load_learner("models/canhealth_model.pkl")

# Load the image
img = load_img("data/test/Septoria/00_septoria_yellowleaf_test.jpg")

# Convert to FastAI's PILImage format
img = PILImage.create(img)

# Get the model predictions
pred_class, pred_idx, outputs = model.predict(img)

print(f"The image is predicted to be '{pred_class}'.")

# from fastai.vision.all import load_learner, PILImage
# from keras_preprocessing.image import load_img
# import pathlib
# from pathlib import Path
# import numpy as np

# # Force FastAI to use WindowsPath instead of PosixPath
# pathlib.PosixPath = pathlib.WindowsPath

# # Load the model
# # model_path = str(Path(r"C:\Users\admin\dataScienceBootcamp\cannabis-health-diagnosis\models\canhealth_model.pkl"))
# # model = load_learner(model_path)
# model = load_learner("models/canhealth_model.pkl")

# # Define the class names
# class_names = [
#     'Aphids',
#     'Botrytis',
#     'Dehydration',
#     'Healthy',
#     'Leaf Miners',
#     'Nitrogen Deficiency',
#     'Nutrient Burn',
#     'Overwatering',
#     'PH Fluctuation',
#     'Phosphorus Deficiency',
#     'Potassium Deficiency',
#     'Powdery Mildew',
#     'Septoria']

# # Load the image
# img = load_img("data/test/Septoria/00_septoria_yellowleaf_test.jpg")

# # Convert to FastAI's PILImage format
# img = PILImage.create(img)

# # Get the model predictions
# pred_class, pred_idx, outputs = model.predict(img)

# # Print the outputs for debugging
# print(f"Outputs: {outputs}")

# # Show the top 3 predicted labels
# top_n = 3 
# sorted_indices = np.argsort(outputs)[-1:-top_n-1:-1]

# print(f"The top {top_n} possible diagnoses are:")
# for i in sorted_indices:
#     print(f"{class_names[i]}: {outputs[i]:.2f} probability")
