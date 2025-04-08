import tensorflow as tf
import numpy as np
import cv2
from tensorflow.keras.preprocessing import image # type: ignore
import matplotlib.pyplot as plt

# Load trained model
model = tf.keras.models.load_model("fire_detection_model.h5")

# Define class labels
class_labels = ["Fire", "Non-Fire"]

def predict_fire(image_path):
    # Load image and preprocess
    img = image.load_img(image_path, target_size=(128, 128))  # Resize to match model input
    img_array = image.img_to_array(img)  # Convert to array
    img_array = np.expand_dims(img_array, axis=0)  # Expand dimensions for batch
    img_array /= 255.0  # Normalize
    
    # Make prediction
    prediction = model.predict(img_array)
    predicted_class = int(prediction[0][0] > 0.5)  # 0: Non-Fire, 1: Fire
    confidence = prediction[0][0]

    print(f"Prediction: {class_labels[predicted_class]} (Confidence: {confidence:.2f})")

    return class_labels[predicted_class]




def show_prediction(image_path):
    result = predict_fire(image_path)
    
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
    
    plt.imshow(img)
    plt.axis("off")
    plt.title(f"Prediction: {result}")
    plt.show()

# Test with an image
show_prediction(r"yeniTest/non-fire\images.jpg")
