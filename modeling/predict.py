import pickle
import numpy as np
from keras_preprocessing.image import load_img, img_to_array

# Load the model
with open("models/canhealth_model.pkl", "rb") as f:
    model = pickle.load(f)

# Define the class names
class_names = [
    'Aphids',
    'Botrytis',
    'Dehydration',
    'Healthy',
    'Leaf Miners',
    'Nitrogen Deficiency',
    'Nutrient Burn',
    'Overwatering',
    'PH Fluctuation',
    'Phosphorus Deficiency',
    'Potassium Deficiency',
    'Powdery Mildew',
    'Septoria']

# Load an image from the test set
img = load_img("data/test/Septoria/00_septoria_yellowleaf_test.jpg", target_size=(224, 224))

# Convert the image to an array
img_array = img_to_array(img)
print(img_array)

print(img_array.shape)

img_array = np.reshape(img_array, (1, 255, 255, 3))

# Get the model predictions
predictions = model.predict(img_array)
print("predictions:", predictions)

# Get the class index with the highest predicted probability
class_index = np.argmax(predictions[0])

# Get the predicted class label
predicted_label = class_names[class_index]

print("The image is predicted to be '{}'.".format(predicted_label))