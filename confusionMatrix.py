import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore
from sklearn.metrics import confusion_matrix, classification_report

# Load trained model
model = tf.keras.models.load_model("fire_detection_model.h5")

# Define dataset path for testing
test_dir = "split_dataset/test"

# Data generator for test set (no augmentation, only rescaling)
test_datagen = ImageDataGenerator(rescale=1.0/255)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(128, 128),
    batch_size=32,
    class_mode='binary',
    shuffle=False  # Important! Prevent shuffling for correct label order
)

# Get true labels
true_labels = test_generator.classes

# Get model predictions (probabilities)
pred_probs = model.predict(test_generator)
pred_labels = (pred_probs > 0.5).astype(int).flatten()  # Convert probabilities to binary labels

# Create confusion matrix
conf_matrix = confusion_matrix(true_labels, pred_labels)

# Plot the confusion matrix
plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=["Non-Fire", "Fire"], yticklabels=["Non-Fire", "Fire"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()


print("Classification Report:\n")
print(classification_report(true_labels, pred_labels, target_names=["Non-Fire", "Fire"]))
