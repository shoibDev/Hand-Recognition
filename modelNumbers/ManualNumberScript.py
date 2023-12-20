import warnings
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

def preprocess_image(image_path):
    # Read the image in grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Resize the image to 28x28 pixels
    resized_image = cv2.resize(image, (28, 28))

    # Normalize the image
    normalized_image = resized_image / 255.0

    # Reshape the image for the CNN
    reshaped_image = normalized_image.reshape(-1, 28, 28, 1)

    return reshaped_image

# Load the saved model
model_path = r'C:\Users\sediq\OneDrive\Desktop\Programming\Coding Languages\pyCode\SentimentPy\modelNumbers\handwritten_recognition_model'
loaded_model = load_model(model_path)

# Path to the input image
image_path = r'C:\Users\sediq\OneDrive\Desktop\Programming\Coding Languages\pyCode\SentimentPy\myName.png'  # Replace with the path to your PNG image

# Preprocess the input image
input_data = preprocess_image(image_path)

# Display the input image
plt.imshow(input_data.reshape(28, 28), cmap='gray')
plt.show()

# Make the prediction
prediction = loaded_model.predict(input_data)

# Get the predicted class
predicted_class = np.argmax(prediction)

print(f"Predicted class for the input image: {predicted_class}")
