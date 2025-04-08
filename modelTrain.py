import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report

# veri setlerinin konumları
train_dir = "split_dataset/train"
val_dir = "split_dataset/val"
test_dir = "split_dataset/test"

# Veri ön işleme ve augmentasyon işlemleri
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

test_datagen = ImageDataGenerator(rescale=1.0 / 255.0)  

# Veri setlerinin yüklenmesi
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(128, 128),
    batch_size=32,
    class_mode='binary'
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(128, 128),
    batch_size=32,
    class_mode='binary'
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(128, 128),
    batch_size=32,
    class_mode='binary',
    shuffle=False  # karmaşıklık matrisinde sıralama bozulmaması için
)

# CNN modelini oluşturma işlemi
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D(2, 2),
    
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')  # Binary sınıflandırma için sigmoid
])

# Modeli derleme işlemi
model.compile(optimizer=Adam(learning_rate=0.001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# modeli eğitme işlemi
epochs = 10
history = model.fit(train_generator, validation_data=val_generator, epochs=epochs)

# modeli kaydetme işlemi
model.save("fire_detection_model.h5")

print("Model training completed and saved!")

# ---------------------------------------
# test verisi ile modelin değerlendirilmesi
# ---------------------------------------

# Doğru etiketleri ve tahmin edilen etiketleri alma
true_labels = test_generator.classes
pred_probs = model.predict(test_generator)
pred_labels = (pred_probs > 0.5).astype(int).flatten()  # Convert probabilities to binary

# Confusion Matrix
conf_matrix = confusion_matrix(true_labels, pred_labels)

# Confsion matrix'i görselleştirme
plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=["Non-Fire", "Fire"], yticklabels=["Non-Fire", "Fire"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# Sınıflandırma raporu hesaplama
print("\nClassification Report:\n")
print(classification_report(true_labels, pred_labels, target_names=["Non-Fire", "Fire"]))
