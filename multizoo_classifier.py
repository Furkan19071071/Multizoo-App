import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms, models
import torchvision.transforms.functional as TF
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, classification_report, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
import timm  
import random
import glob
from tqdm import tqdm
import seaborn as sns
from torch.optim.lr_scheduler import OneCycleLR

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

BATCH_SIZE = 16
NUM_EPOCHS = 30
IMAGE_SIZE = 224 
MODEL_NAME = "vit_base_patch16_224" 
LEARNING_RATE = 3e-4

DATASET_PATH = "train"  
TRAIN_PATH = os.path.join(DATASET_PATH, "train")  
MODEL_SAVE_PATH = "multizoo_transformer_model.pth"

classes = sorted([dirname for dirname in os.listdir(TRAIN_PATH) if os.path.isdir(os.path.join(TRAIN_PATH, dirname))])
num_classes = len(classes)
print(f"Total {num_classes} classes: {classes}")

train_transforms = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.2),  # Dikey çevirme ekleyin
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomAffine(degrees=20, translate=(0.1, 0.1), scale=(0.8, 1.2)),  # Ölçeklendirme ekleyin
    transforms.RandomPerspective(distortion_scale=0.2, p=0.5),  # Perspektif değişiklikleri
    transforms.RandomGrayscale(p=0.1),  # Bazen gri tonlamaya dönüştürme
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transforms = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class MultiZooDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = sorted([dirname for dirname in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, dirname))])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        self.image_paths = []
        self.labels = []
        
        for class_name in self.classes:
            class_dir = os.path.join(root_dir, class_name)
            for img_name in os.listdir(class_dir):
                if img_name.endswith(('.jpg', '.jpeg', '.png')):
                    self.image_paths.append(os.path.join(class_dir, img_name))
                    self.labels.append(self.class_to_idx[class_name])
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

def analyze_validation_performance(model, data_loader, classes, device):
    """
    Analyze performance on validation set and return detailed metrics.
    
    Args:
        model: Trained model
        data_loader: Validation data loader
        classes: List of class labels
        device: Processing device (CPU/GPU)
    
    Returns:
        accuracy, precision, recall, f1 values
    """
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(data_loader, desc="Validation"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=classes, zero_division=0))
    
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.show()
    
    return accuracy, precision, recall, f1

def predict_image(model, image_path, transform, device, classes):
    model.eval()
    
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted_idx = torch.max(probabilities, 1)
    
    predicted_class = classes[predicted_idx.item()]
    confidence_score = confidence.item()
    
    return predicted_class, confidence_score

if __name__ == '__main__':
    full_dataset = MultiZooDataset(TRAIN_PATH, transform=train_transforms)

    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    val_dataset.dataset.transform = val_transforms

    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    def create_model(num_classes):
        model = timm.create_model(
            MODEL_NAME, 
            pretrained=True, 
            num_classes=num_classes,
            drop_rate=0.2,          
            drop_path_rate=0.1      
        )
        return model

    model = create_model(num_classes)
    model = model.to(device)

    
    train_labels = [train_dataset[i][1] for i in range(len(train_dataset))]
    unique_labels = np.unique(train_labels)
    class_weights = compute_class_weight('balanced', classes=unique_labels, y=train_labels)
    class_weights = torch.FloatTensor(class_weights).to(device)
    
    print("Sınıf ağırlıkları:", class_weights.cpu().numpy())
    
    
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

   
    scheduler = OneCycleLR(
        optimizer,
        max_lr=LEARNING_RATE,
        steps_per_epoch=len(train_loader),
        epochs=NUM_EPOCHS,
        pct_start=0.3, 
        div_factor=25, 
        final_div_factor=1000,  
    )

    
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []

   
    best_val_loss = float('inf')
    patience = 5
    early_stop_counter = 0

   
    def train_one_epoch(model, loader, criterion, optimizer, device):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in tqdm(loader, desc="Training"):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        epoch_loss = running_loss / len(loader.dataset)
        epoch_acc = correct / total
        
        return epoch_loss, epoch_acc

    
    def validate(model, loader, criterion, device):
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for inputs, labels in tqdm(loader, desc="Validation"):
                inputs, labels = inputs.to(device), labels.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                running_loss += loss.item() * inputs.size(0)
                
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        epoch_loss = running_loss / len(loader.dataset)
        epoch_acc = correct / total
        
        
        precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
        recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
        f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
        
        return epoch_loss, epoch_acc, precision, recall, f1

    
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    val_precisions = []
    val_recalls = []
    val_f1s = []
    lr_values = []

    print("Starting training...")

    for epoch in range(NUM_EPOCHS):
       
        current_lr = optimizer.param_groups[0]['lr']
        lr_values.append(current_lr)
        
       
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
       
        val_loss, val_acc, precision, recall, f1 = validate(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        val_precisions.append(precision)
        val_recalls.append(recall)
        val_f1s.append(f1)
        

        
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}")
        print(f"Learning Rate: {current_lr:.6f}")
        print(f"Training Loss: {train_loss:.4f}, Training Accuracy: {train_acc:.4f}")
        print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}")
        print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}")
        
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            best_epoch = epoch
            early_stop_counter = 0
           
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_acc': val_acc,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'classes': classes
            }, MODEL_SAVE_PATH)
            print(f"Model saved: {MODEL_SAVE_PATH}")
        else:
            early_stop_counter += 1
            if early_stop_counter >= patience:
                print(f"Early stopping triggered after {patience} epochs.")
                break

    print("Training completed!")

    plt.figure(figsize=(15, 10))

  
    plt.subplot(2, 2, 1)
    plt.plot(train_losses, 'b-', label='Training Loss')
    plt.plot(val_losses, 'r-', label='Validation Loss')
    plt.axvline(x=best_epoch, color='g', linestyle='--', label=f'Best Model (Epoch {best_epoch+1})')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)

   
    plt.subplot(2, 2, 2)
    plt.plot(train_accs, 'b-', label='Training Accuracy')
    plt.plot(val_accs, 'r-', label='Validation Accuracy')
    plt.axvline(x=best_epoch, color='g', linestyle='--', label=f'Best Model (Epoch {best_epoch+1})')
    plt.axhline(y=0.65, color='k', linestyle='--', label='Target Accuracy (65%)')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.grid(True)

    
    plt.subplot(2, 2, 3)
    plt.plot(val_precisions, 'r-', label='Precision')
    plt.plot(val_recalls, 'g-', label='Recall')
    plt.plot(val_f1s, 'b-', label='F1-Score')
    plt.axvline(x=best_epoch, color='g', linestyle='--', label=f'Best Model (Epoch {best_epoch+1})')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.title('Validation Metrics')
    plt.legend()
    plt.grid(True)

    
    plt.subplot(2, 2, 4)
    plt.plot(lr_values, 'k-')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Change')
    plt.yscale('log')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('learning_curves.png', dpi=300, bbox_inches='tight')
    plt.show()

   
    checkpoint = torch.load(MODEL_SAVE_PATH)
    model.load_state_dict(checkpoint['model_state_dict'])
    best_epoch = checkpoint['epoch']
    print(f"Best model loaded:")
    print(f"Epoch: {best_epoch+1}")
    print(f"Validation Accuracy: {checkpoint['val_acc']:.4f}")
    print(f"Precision: {checkpoint['precision']:.4f}")
    print(f"Recall: {checkpoint['recall']:.4f}")
    print(f"F1-Score: {checkpoint['f1']:.4f}")

  
    print("\nValidation performance analysis for best model:")
    val_acc, val_precision, val_recall, val_f1 = analyze_validation_performance(model, val_loader, classes, device)

    
    results_df = pd.DataFrame({
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
        'Value': [val_acc, val_precision, val_recall, val_f1]
    })

    print("\nSummary Performance Table:")
    print(results_df)