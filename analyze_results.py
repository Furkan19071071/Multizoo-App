import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.manifold import TSNE
import torch.nn.functional as F
from multizoo_classifier import MultiZooDataset, val_transforms, analyze_validation_performance

# Yüklenecek model ve veri seti yolu
MODEL_PATH = "multizoo_transformer_model.pth"
DATASET_PATH = "train/train"  # Veri setinin bulunduğu yol

# Sınıf dağılımını analiz etme
def analyze_class_distribution(dataset_path):
    """Veri setindeki sınıf dağılımını analiz eder."""
    import os
    
    classes = sorted([dirname for dirname in os.listdir(dataset_path) 
                      if os.path.isdir(os.path.join(dataset_path, dirname))])
    
    class_counts = []
    for cls in classes:
        class_path = os.path.join(dataset_path, cls)
        num_samples = len([name for name in os.listdir(class_path) 
                          if os.path.isfile(os.path.join(class_path, name))])
        class_counts.append(num_samples)
    
    # Sınıf dağılımını görselleştir
    plt.figure(figsize=(12, 8))
    sns.barplot(x=classes, y=class_counts)
    plt.title('Veri Setindeki Sınıf Dağılımı')
    plt.xlabel('Sınıflar')
    plt.ylabel('Örnek Sayısı')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig('class_distribution.png')
    plt.show()
    
    return classes, class_counts

# En çok yanlış sınıflandırılan örnekleri bulma
def find_misclassified_examples(model, data_loader, device, top_n=10):
    """En çok yanlış sınıflandırılan örnekleri bulur."""
    model.eval()
    misclassified = []
    
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            probas = F.softmax(outputs, dim=1)
            
            _, preds = torch.max(outputs, 1)
            incorrect_mask = (preds != labels)
            
            if incorrect_mask.sum().item() > 0:
                incorrect_inputs = inputs[incorrect_mask]
                incorrect_labels = labels[incorrect_mask]
                incorrect_preds = preds[incorrect_mask]
                incorrect_probas = probas[incorrect_mask]
                
                for i in range(len(incorrect_inputs)):
                    confidence = incorrect_probas[i, incorrect_preds[i]].item()
                    misclassified.append({
                        'input': incorrect_inputs[i].cpu(),
                        'true_label': incorrect_labels[i].item(),
                        'pred_label': incorrect_preds[i].item(),
                        'confidence': confidence
                    })
    
    # Güvene göre sırala (en düşükten en yükseğe)
    misclassified.sort(key=lambda x: x['confidence'])
    
    return misclassified[:top_n]

# t-SNE ile özellik görselleştirmesi
def visualize_features(model, data_loader, device, num_samples=1000):
    """t-SNE kullanarak özellik temsillerini görselleştirir."""
    model.eval()
    features = []
    labels = []
    count = 0
    
    # Orta bir katmandan özellik çıkarma
    def get_features_hook(module, input, output):
        nonlocal features
        features.append(output.cpu().numpy())
    
    # Model mimarisine bağlı olarak uygun katmanı seçin
    # ViT için genellikle son katmandan önceki katman
    for name, module in model.named_modules():
        if "blocks" in name and ".11" in name:  # Son bloktan önceki blok
            handle = module.register_forward_hook(get_features_hook)
            break
    
    with torch.no_grad():
        for inputs, targets in data_loader:
            if count >= num_samples:
                break
                
            inputs = inputs.to(device)
            batch_size = inputs.size(0)
            
            _ = model(inputs)
            
            labels.append(targets.numpy())
            count += batch_size
            
            if count >= num_samples:
                break
    
    handle.remove()
    
    # Özellikleri birleştir
    features = np.vstack([f for f in features])
    features = features.reshape(features.shape[0], -1)  # Düzleştir
    
    # Örnekleme sayısına sınırla
    features = features[:num_samples]
    labels = np.concatenate(labels)[:num_samples]
    
    # t-SNE uygula
    tsne = TSNE(n_components=2, random_state=42)
    features_2d = tsne.fit_transform(features)
    
    # Görselleştir
    plt.figure(figsize=(10, 8))
    for i, cls in enumerate(np.unique(labels)):
        plt.scatter(
            features_2d[labels == cls, 0],
            features_2d[labels == cls, 1],
            label=f'Sınıf {cls}',
            alpha=0.7
        )
    plt.title('t-SNE Özellik Görselleştirmesi')
    plt.xlabel('t-SNE Bileşen 1')
    plt.ylabel('t-SNE Bileşen 2')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('tsne_visualization.png')
    plt.show()

if __name__ == "__main__":
    # CUDA kontrolü
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Kullanılan cihaz: {device}")
    
    # Modeli yükle
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    
    # MultiZooDataset'i kullanarak veri setini yükle
    # Bu aşamada ihtiyacınıza göre MultiZooDataset sınıfını import etmeniz gerekecektir
    full_dataset = MultiZooDataset(DATASET_PATH, transform=val_transforms)
    classes = checkpoint['classes']
    
    print(f"Sınıflar: {classes}")
    
    # Sınıf dağılımını analiz et
    print("Sınıf dağılımını analiz ediliyor...")
    analyze_class_distribution(DATASET_PATH)
    
    print("Detaylı performans analizi başarıyla tamamlandı!")
