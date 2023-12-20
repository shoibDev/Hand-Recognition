import warnings
from matplotlib import MatplotlibDeprecationWarning

warnings.filterwarnings("ignore", category=MatplotlibDeprecationWarning)

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import tensorflow as tf
from emnist import extract_test_samples

# Load the saved model
model_path = r'C:\Users\sediq\OneDrive\Desktop\Programming\Coding Languages\pyCode\SentimentPy\modelWords\handwritten_letters_recognition_model'

loaded_model = load_model(model_path)

# Load the EMNIST dataset (letters)
test_images, test_labels = extract_test_samples('letters')

# Normalize the images
test_images = test_images / 255.0

# Reshape the images for the CNN
test_images = test_images.reshape(-1, 28, 28, 1)

# Number of random images to display
num_images = 5  # You can change this number to display more or less random images

letters = ' ABCDEFGHIJKLMNOPQRSTUVWXYZ'  # Note the added space at the beginning

# Loop through the random image indices, make predictions, and display the images
for i in range(num_images):
    # Generate a random index
    idx = np.random.randint(0, test_images.shape[0])

    # Load a sample image from the test set
    sample_image = test_images[idx]

    # Display the sample image
    plt.imshow(sample_image.reshape(28, 28), cmap='gray')
    plt.show()

    # Prepare the input for prediction (expand dimensions to match the expected input shape)
    input_data = np.expand_dims(sample_image, axis=0)

    # Make the prediction
    prediction = loaded_model.predict(input_data)

    # Get the predicted class
    predicted_class = np.argmax(prediction)

    # Convert the predicted class to a letter
    predicted_letter = letters[predicted_class]

    print(f"Predicted letter for image {i + 1}: {predicted_letter}")