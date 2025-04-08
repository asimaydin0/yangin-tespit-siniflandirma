import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import os
import numpy as np
from datetime import datetime
import tensorflow as tf

class ModernFireDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Yangın Tespit Uygulaması")
        self.root.geometry("900x600")
        self.root.configure(bg="#f0f0f0")
        self.root.resizable(True, True)
        
        # Değişkenler
        self.image_path = None
        self.original_image = None
        self.displayed_image = None
        
        # TensorFlow modelini yükle
        try:
            self.model = tf.keras.models.load_model("fire_detection_model.h5")
            self.model_loaded = True
        except Exception as e:
            self.model_loaded = False
            print(f"Model yüklenirken hata: {str(e)}")
        
        # Resim boyutu (eğitimle aynı olmalı)
        self.IMAGE_SIZE = (128, 128)
        
        # Ana çerçeve
        self.main_frame = tk.Frame(self.root, bg="#f0f0f0")
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Başlık
        self.title_label = tk.Label(
            self.main_frame, 
            text="Yangın Tespit Uygulaması", 
            font=("Helvetica", 24, "bold"),
            fg="#e74c3c",  # Kırmızımsı renk
            bg="#f0f0f0"
        )
        self.title_label.pack(pady=(0, 20))
        
        # İçerik çerçevesi
        self.content_frame = tk.Frame(self.main_frame, bg="#ffffff", bd=2, relief=tk.GROOVE)
        self.content_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Sol panel (resim gösterimi)
        self.image_panel = tk.Frame(self.content_frame, bg="#ffffff")
        self.image_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Resim etiketi
        self.image_container = tk.Label(
            self.image_panel, 
            text="Resim burada gösterilecek", 
            font=("Helvetica", 12),
            bg="#f5f5f5",
            relief=tk.RIDGE,
            width=50, 
            height=20
        )
        self.image_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Sağ panel (kontroller)
        self.control_panel = tk.Frame(self.content_frame, bg="#ffffff", width=200)
        self.control_panel.pack(side=tk.RIGHT, fill=tk.BOTH, padx=10, pady=10)
        
        # Resim yükleme butonu
        self.upload_button = tk.Button(
            self.control_panel,
            text="Resim Yükle",
            font=("Helvetica", 12),
            bg="#3498db",
            fg="white",
            padx=20,
            pady=10,
            cursor="hand2",
            command=self.upload_image
        )
        self.upload_button.pack(fill=tk.X, pady=10)
        
        # Analiz butonu
        self.analyze_button = tk.Button(
            self.control_panel,
            text="Yangın Analizi Yap",
            font=("Helvetica", 12),
            bg="#2ecc71",
            fg="white",
            padx=20,
            pady=10,
            cursor="hand2",
            state=tk.DISABLED,
            command=self.analyze_image
        )
        self.analyze_button.pack(fill=tk.X, pady=10)
        
        # Sonuç kutusu
        self.result_frame = tk.LabelFrame(
            self.control_panel, 
            text="Analiz Sonucu", 
            font=("Helvetica", 12, "bold"),
            bg="#ffffff",
            fg="#333333",
            padx=10, 
            pady=10
        )
        self.result_frame.pack(fill=tk.X, pady=20)
        
        self.result_label = tk.Label(
            self.result_frame, 
            text="Henüz analiz yapılmadı", 
            font=("Helvetica", 10),
            bg="#ffffff",
            fg="#555555",
            wraplength=180,
            justify=tk.LEFT
        )
        self.result_label.pack(fill=tk.X, pady=5)
        
        # Resim bilgisi
        self.info_frame = tk.LabelFrame(
            self.control_panel, 
            text="Resim Bilgisi", 
            font=("Helvetica", 12, "bold"),
            bg="#ffffff",
            fg="#333333",
            padx=10, 
            pady=10
        )
        self.info_frame.pack(fill=tk.X, pady=10)
        
        self.info_label = tk.Label(
            self.info_frame, 
            text="Henüz resim yüklenmedi", 
            font=("Helvetica", 10),
            bg="#ffffff",
            fg="#555555",
            wraplength=180,
            justify=tk.LEFT
        )
        self.info_label.pack(fill=tk.X, pady=5)
        
        # Model durum bilgisi
        self.model_info_label = tk.Label(
            self.control_panel,
            text=f"Model Durumu: {'Yüklendi ✓' if self.model_loaded else 'Yüklenemedi ✗'}",
            font=("Helvetica", 9),
            bg="#ffffff",
            fg="#27ae60" if self.model_loaded else "#e74c3c"
        )
        self.model_info_label.pack(pady=(10, 0), anchor=tk.W)
        
        # Durum çubuğu
        self.status_bar = tk.Label(
            self.root, 
            text="Hazır", 
            bd=1, 
            relief=tk.SUNKEN, 
            anchor=tk.W, 
            bg="#f0f0f0",
            fg="#555555",
            padx=10
        )
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
    
    def upload_image(self):
        """Resim yükleme işlemi"""
        self.image_path = filedialog.askopenfilename(
            title="Resim Seçin",
            filetypes=[
                ("Resim Dosyaları", "*.jpg *.jpeg *.png *.bmp"),
                ("Tüm Dosyalar", "*.*")
            ]
        )
        
        if self.image_path:
            try:
                # Resmi yükle ve görüntüle
                self.original_image = Image.open(self.image_path)
                self.display_image(self.original_image)
                
                # Resim bilgisini güncelle
                img_size = os.path.getsize(self.image_path) / 1024  # KB cinsinden
                img_width, img_height = self.original_image.size
                file_name = os.path.basename(self.image_path)
                
                info_text = f"Dosya adı: {file_name}\n"
                info_text += f"Boyutlar: {img_width}x{img_height} px\n"
                info_text += f"Dosya boyutu: {img_size:.1f} KB"
                
                self.info_label.config(text=info_text)
                
                # Analiz butonunu aktifleştir
                self.analyze_button.config(state=tk.NORMAL)
                
                # Durum çubuğunu güncelle
                self.status_bar.config(text=f"Resim yüklendi: {self.image_path}")
                
                # Sonuç etiketini sıfırla
                self.result_label.config(text="Henüz analiz yapılmadı")
                
            except Exception as e:
                messagebox.showerror("Hata", f"Resim yüklenirken bir hata oluştu: {str(e)}")
    
    def display_image(self, img):
        """Resmi arayüzde gösterme işlemi"""
        # Resmi uygun boyuta getir
        container_width = self.image_container.winfo_width()
        container_height = self.image_container.winfo_height()
        
        # Eğer container henüz render edilmediyse varsayılan değerler kullan
        if container_width <= 1:
            container_width = 400
        if container_height <= 1:
            container_height = 300
        
        # Resmi en-boy oranını koruyarak yeniden boyutlandır
        img_width, img_height = img.size
        ratio = min(container_width/img_width, container_height/img_height)
        new_width = int(img_width * ratio)
        new_height = int(img_height * ratio)
        
        resized_img = img.resize((new_width, new_height), Image.LANCZOS)
        
        # Tkinter'da göstermek için PhotoImage'e dönüştür
        self.displayed_image = ImageTk.PhotoImage(resized_img)
        
        # Resmi göster
        self.image_container.config(image=self.displayed_image, text="")
    
    def preprocess_image(self, img):
        """Resmi model için hazırlama"""
        img_resized = img.resize(self.IMAGE_SIZE)
        img_array = np.array(img_resized) / 255.0  # normalizasyon
        return np.expand_dims(img_array, axis=0)
    
    def analyze_image(self):
        """Resmi TensorFlow modeli ile analiz etme işlemi"""
        if not self.original_image:
            messagebox.showwarning("Uyarı", "Lütfen önce bir resim yükleyin!")
            return
        
        if not self.model_loaded:
            # Eğer model yüklenemediyse basit renk analizi yap
            self.analyze_with_color()
            return
        
        try:
            # Durum çubuğunu güncelle
            self.status_bar.config(text="TensorFlow ile analiz yapılıyor...")
            self.root.update()
            
            # Resmi modele uygun formata getir
            img_rgb = self.original_image.convert('RGB')
            img_array = self.preprocess_image(img_rgb)
            
            # Tahmin yap
            prediction = self.model.predict(img_array)[0][0]
            confidence = prediction * 100
            
            # Debug bilgisi
            print(f"Model tahmini: {prediction}")
            
            # Sonucu göster
            if prediction < 0.5:
                result = f"🔥 YANGIN TESPİT EDİLDİ!\nGüven Oranı: %{100 - confidence:.2f}"
                result_color = "#e74c3c"  # Kırmızı
                is_fire = True
            else:
                result = f"✅ Yangın tespit edilmedi.\nGüven Oranı: %{confidence:.2f}"
                result_color = "#2ecc71"  # Yeşil
                is_fire = False
                
            # Analiz sonucunu göster
            self.result_label.config(text=result, fg=result_color)
            
            # Log dosyasına kaydet
            self._log_result(is_fire, confidence if is_fire else 100-confidence)
            
            # Durum çubuğunu güncelle
            self.status_bar.config(text="TensorFlow analizi tamamlandı")
            
        except Exception as e:
            messagebox.showerror("Hata", f"TensorFlow analizi sırasında hata: {str(e)}")
            self.status_bar.config(text="Hata: TensorFlow analizi başarısız oldu")
            # Hata durumunda renk analizi dene
            self.analyze_with_color()
    
    def analyze_with_color(self):
        """Renk analizi ile basit yangın tespiti yapar (yedek metod)"""
        try:
            # Durum çubuğunu güncelle
            self.status_bar.config(text="Renk analizi yapılıyor...")
            self.root.update()
            
            # Resmi numpy dizisine dönüştür
            img_array = np.array(self.original_image)
            
            # Kırmızı renk yoğunluğunu kontrol et
            if len(img_array.shape) == 3:  # RGB resim
                # Kırmızı kanal değerlerini al
                red_channel = img_array[:, :, 0]
                
                # Kırmızı yoğunluğu hesapla
                red_intensity = np.mean(red_channel)
                
                # Sarı-turuncu-kırmızı renk oranını hesapla
                red_pixels = np.sum((img_array[:, :, 0] > 200) & 
                                   (img_array[:, :, 1] < 150) & 
                                   (img_array[:, :, 2] < 150))
                
                orange_pixels = np.sum((img_array[:, :, 0] > 220) & 
                                      (img_array[:, :, 1] > 110) & 
                                      (img_array[:, :, 1] < 180) & 
                                      (img_array[:, :, 2] < 100))
                
                yellow_pixels = np.sum((img_array[:, :, 0] > 220) & 
                                      (img_array[:, :, 1] > 200) & 
                                      (img_array[:, :, 2] < 100))
                
                total_pixels = img_array.shape[0] * img_array.shape[1]
                fire_color_ratio = (red_pixels + orange_pixels + yellow_pixels) / total_pixels
                
                # Karar verme
                if fire_color_ratio > 0.05:
                    probability = min(fire_color_ratio * 10, 0.99)
                    result = f"🔥 YANGIN TESPİT EDİLDİ! (Renk Analizi)\nGüven Oranı: %{probability*100:.1f}"
                    result_color = "#e74c3c"  # Kırmızı
                    is_fire = True
                else:
                    result = "✅ Yangın tespit edilmedi. (Renk Analizi)\nBu resimde yangın belirtisi bulunmuyor."
                    result_color = "#2ecc71"  # Yeşil
                    is_fire = False
                    probability = 0
                
                # Analiz sonucunu göster
                self.result_label.config(text=result, fg=result_color)
                
                # Log dosyasına kaydet
                self._log_result(is_fire, probability*100)
                
            else:
                # Gri tonlamalı veya farklı formattaki resimler için
                self.result_label.config(
                    text="Bu resim formatı analiz için uygun değil.\nLütfen renkli (RGB) bir resim yükleyin.",
                    fg="#e67e22"  # Turuncu
                )
            
            # Durum çubuğunu güncelle
            self.status_bar.config(text="Renk analizi tamamlandı")
            
        except Exception as e:
            messagebox.showerror("Hata", f"Renk analizi sırasında bir hata oluştu: {str(e)}")
            self.status_bar.config(text="Hata: Renk analizi başarısız oldu")
    
    def _log_result(self, is_fire, confidence):
        """Analiz sonucunu log dosyasına kaydet"""
        try:
            log_dir = "logs"
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)
            
            log_file = os.path.join(log_dir, "fire_detection_log.txt")
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(f"[{timestamp}] Resim: {os.path.basename(self.image_path)}, ")
                f.write(f"Yangın Tespit: {'Evet' if is_fire else 'Hayır'}, ")
                f.write(f"Güven Oranı: %{confidence:.1f}\n")
                
        except Exception as e:
            print(f"Log kaydederken hata: {str(e)}")

# Uygulamayı başlat
if __name__ == "__main__":
    root = tk.Tk()
    app = ModernFireDetectionApp(root)
    
    # Pencere kapanırken yapılacaklar
    def on_closing():
        if messagebox.askokcancel("Çıkış", "Uygulamadan çıkmak istediğinize emin misiniz?"):
            root.destroy()
    
    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()