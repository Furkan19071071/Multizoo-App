import sys
import os
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
import timm
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Sabit değişkenler
MODEL_PATH = "multizoo_transformer_model.pth"
IMAGE_SIZE = 224  # ViT için standart boyut
MODEL_NAME = "vit_base_patch16_224"  # Vision Transformer modeli
RESULT_FOLDER = "results"  # Sonuç görselleri için klasör

# Sonuç klasörünü oluştur
os.makedirs(RESULT_FOLDER, exist_ok=True)

# Görüntü dönüştürme
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Cihaz seçimi
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model sınıfını oluştur
def load_model(model_path):
    # Checkpoint yükle
    checkpoint = torch.load(model_path, map_location=device)
    classes = checkpoint['classes']
    num_classes = len(classes)
    
    # Modeli oluştur
    model = timm.create_model(MODEL_NAME, pretrained=False, num_classes=num_classes)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    return model, classes

# Tahmin fonksiyonu
def predict_image(model, image_path, transform, device, classes):
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        # En yüksek olasılığı ve indeksini al
        confidence, predicted_idx = torch.max(probabilities, 1)
        
        # Tüm sınıflar için olasılıkları al
        all_probs = probabilities.squeeze().cpu().numpy()
    
    predicted_class = classes[predicted_idx.item()]
    confidence_score = confidence.item()
    
    return predicted_class, confidence_score, all_probs

# Ana uygulama sınıfı
class MultiZooApp:
    def __init__(self, root):
        self.root = root
        self.root.title("MultiZoo Hayvan Sınıflandırma")
        self.root.geometry("900x700")
        self.root.configure(bg="#f0f0f0")
        
        self.setup_ui()
        
        # Model yüklenmeye çalışılıyor
        try:
            self.model, self.classes = load_model(MODEL_PATH)
            self.status_label.config(text=f"Model yüklendi. {len(self.classes)} sınıf bulundu.")
        except Exception as e:
            self.status_label.config(text=f"Model yüklenemedi: {e}")
            messagebox.showerror("Hata", f"Model yüklenemedi: {e}")
    
    def setup_ui(self):
        # Ana çerçeve
        main_frame = ttk.Frame(self.root, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Stil tanımlamaları
        style = ttk.Style()
        style.configure("TButton", font=("Arial", 12), padding=10)
        style.configure("TLabel", font=("Arial", 12), background="#f0f0f0")
        style.configure("Header.TLabel", font=("Arial", 16, "bold"), background="#f0f0f0")
        
        # Başlık
        header_label = ttk.Label(main_frame, text="MultiZoo Hayvan Sınıflandırma", style="Header.TLabel")
        header_label.pack(pady=10)
        
        # Görüntü yükleme butonu
        self.load_button = ttk.Button(main_frame, text="Görüntü Yükle", command=self.load_image)
        self.load_button.pack(pady=10)
        
        # Görüntü çerçevesi
        self.image_frame = ttk.LabelFrame(main_frame, text="Yüklenen Görüntü")
        self.image_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        self.image_label = ttk.Label(self.image_frame)
        self.image_label.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Sonuç çerçevesi
        result_frame = ttk.LabelFrame(main_frame, text="Tahmin Sonucu")
        result_frame.pack(fill=tk.X, pady=10)
        
        # Sonuç etiketi
        self.result_label = ttk.Label(result_frame, text="Henüz bir görüntü yüklemediniz")
        self.result_label.pack(pady=10)
        
        # Güven skoru etiketi
        self.confidence_label = ttk.Label(result_frame, text="")
        self.confidence_label.pack(pady=5)
        
        # Grafik çerçevesi
        self.chart_frame = ttk.Frame(main_frame)
        self.chart_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Durum çubuğu
        self.status_label = ttk.Label(self.root, text="Durum: Model yükleniyor...", relief=tk.SUNKEN, anchor=tk.W)
        self.status_label.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Görüntü değişkeni
        self.image_path = None
        self.tk_image = None
        
    def load_image(self):
        # Dosya seçiciyi aç
        file_path = filedialog.askopenfilename(
            title="Bir görüntü seç",
            filetypes=[("Görüntü Dosyaları", "*.jpg *.jpeg *.png")]
        )
        
        if file_path:
            try:
                # Görüntüyü arayüzde göster
                self.image_path = file_path
                image = Image.open(file_path)
                
                # Resmi UI'da gösterecek boyuta ayarla
                image.thumbnail((400, 400))
                self.tk_image = ImageTk.PhotoImage(image)
                self.image_label.config(image=self.tk_image)
                
                # Tahmin yap
                self.predict()
            except Exception as e:
                messagebox.showerror("Hata", f"Görüntü yüklenemedi: {e}")
    
    def predict(self):
        if not hasattr(self, 'model'):
            messagebox.showerror("Hata", "Model yüklenemedi!")
            return
        
        if not self.image_path:
            messagebox.showerror("Hata", "Önce bir görüntü yükleyin!")
            return
        
        try:
            self.status_label.config(text="Tahmin yapılıyor...")
            predicted_class, confidence, all_probs = predict_image(
                self.model, self.image_path, transform, device, self.classes
            )
            
            # Sonuçları göster
            self.result_label.config(text=f"Tahmin: {predicted_class}")
            self.confidence_label.config(text=f"Güven Skoru: {confidence * 100:.2f}%")
            
            # Grafik göster
            self.show_probability_chart(all_probs)
            
            self.status_label.config(text=f"Tahmin tamamlandı: {predicted_class}")
        except Exception as e:
            messagebox.showerror("Hata", f"Tahmin yapılırken hata oluştu: {e}")
            self.status_label.config(text=f"Hata: {e}")
    
    def show_probability_chart(self, probabilities):
        # Önceki grafiği temizle
        for widget in self.chart_frame.winfo_children():
            widget.destroy()
        
        # En yüksek 5 olasılığı göster
        top_indices = np.argsort(probabilities)[-5:][::-1]
        top_probs = probabilities[top_indices]
        top_classes = [self.classes[i] for i in top_indices]
        
        # Grafik oluştur
        fig = Figure(figsize=(7, 3), dpi=100)
        ax = fig.add_subplot(111)
        
        # Yatay bar grafiği
        bars = ax.barh(top_classes, top_probs, color='skyblue')
        
        # Yüzde etiketleri ekle
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax.text(width + 0.01, bar.get_y() + bar.get_height()/2, f'{width*100:.1f}%', 
                    ha='left', va='center')
        
        ax.set_xlabel('Olasılık')
        ax.set_title('En Yüksek 5 Tahmin')
        ax.set_xlim(0, 1.1)
        
        # Grafiği UI'a ekle
        canvas = FigureCanvasTkAgg(fig, master=self.chart_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

# Ana fonksiyon
def main():
    root = tk.Tk()
    app = MultiZooApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()