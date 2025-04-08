import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf

# Load the trained model
model = tf.keras.models.load_model("fire_detection_model.h5")
IMAGE_SIZE = (128, 128)  # same as training

# Function to preprocess the image
def preprocess_image(img_path):
    img = Image.open(img_path).convert('RGB')
    img = img.resize(IMAGE_SIZE)
    img_array = np.array(img) / 255.0  # normalization
    return np.expand_dims(img_array, axis=0)

# Predict function
def predict_image():
    if not app.image_path:
        result_label.config(text="No image selected.")
        return
    img_array = preprocess_image(app.image_path)
    prediction = model.predict(img_array)[0][0]
    print(model.predict(img_array))  # Debugging line
    print(f"Prediction: {prediction:.2f}")  # Debugging line
    confidence = prediction * 100

    if prediction < 0.5:
        result = f"🔥 Fire Detected! (Confidence: {100 - confidence:.2f}%)"
        result_label.config(fg="red")
    else:
        result = f"✅ No Fire (Confidence: {confidence:.2f}%)"
        result_label.config(fg="green")

    result_label.config(text=result)

# Select image from file dialog
def browse_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
    if file_path:
        app.image_path = file_path
        img = Image.open(file_path)
        img.thumbnail((300, 300))
        img_tk = ImageTk.PhotoImage(img)
        image_label.config(image=img_tk)
        image_label.image = img_tk
        result_label.config(text="", fg="black")

# Build the UI
app = tk.Tk()
app.title("🔥 Fire Detection UI")
app.geometry("400x500")
app.image_path = None

# UI Widgets
browse_btn = tk.Button(app, text="Choose Image", command=browse_image)
browse_btn.pack(pady=10)

image_label = tk.Label(app)
image_label.pack()

predict_btn = tk.Button(app, text="Predict", command=predict_image)
predict_btn.pack(pady=10)

result_label = tk.Label(app, text="", font=("Arial", 14))
result_label.pack(pady=20)

app.mainloop()
