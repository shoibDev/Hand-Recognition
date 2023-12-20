import warnings
import random
from matplotlib import MatplotlibDeprecationWarning

warnings.filterwarnings("ignore", category=MatplotlibDeprecationWarning)

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import tensorflow as tf

# Load the saved model
model_path = r'C:\Users\sediq\OneDrive\Desktop\Programming\Coding Languages\pyCode\SentimentPy\modelNumbers\handwritten_recognition_model'
loaded_model = load_model(model_path)

# Load the MNIST dataset
mnist = tf.keras.datasets.mnist
(_, _), (test_images, _) = mnist.load_data()

# Normalize the images
test_images = test_images / 255.0

# Reshape the images for the CNN
test_images = test_images.reshape(-1, 28, 28, 1)

# List of indices of images to test
num_samples = 5  # Number of random samples you want to test
image_indices = random.sample(range(len(test_images)), num_samples)

# Loop through the image indices, make predictions, and display the images
for idx in image_indices:
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

    print(f"Predicted class for image {idx}: {predicted_class}")
