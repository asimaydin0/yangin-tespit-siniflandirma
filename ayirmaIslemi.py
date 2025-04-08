import os
import shutil
import random

# Konumları tanımlama 
dataset_path = "dataset/"  # Orijinal veri seti konumu
output_path = "split_dataset/"  # ayrılmış veri seti konumu
categories = ["fire", "non_fire"]  # sınıf etiketleri

# Bölme oranları
# 70% train, 20% validation, 10% test
train_split = 0.7
val_split = 0.2
test_split = 0.1

# Direktörleri oluşturma
for category in categories:
    for folder in ["train", "val", "test"]:
        os.makedirs(os.path.join(output_path, folder, category), exist_ok=True)

# Bölme ve taşıma işlemi fonksiyonu
def split_and_move(category):
    files = os.listdir(os.path.join(dataset_path, category))
    random.shuffle(files)
    
    total = len(files)
    train_count = int(total * train_split)
    val_count = int(total * val_split)
    
    train_files = files[:train_count]
    val_files = files[train_count:train_count + val_count]
    test_files = files[train_count + val_count:]

    for f in train_files:
        shutil.move(os.path.join(dataset_path, category, f), os.path.join(output_path, "train", category, f))
    for f in val_files:
        shutil.move(os.path.join(dataset_path, category, f), os.path.join(output_path, "val", category, f))
    for f in test_files:
        shutil.move(os.path.join(dataset_path, category, f), os.path.join(output_path, "test", category, f))

# Fonkksiyonu her kategori için çağırma
for category in categories:
    split_and_move(category)

print("Dataset successfully split into train, val, and test sets!")
