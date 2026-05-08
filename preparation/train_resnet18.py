import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast # NEU: Für massiv schnelleres GPU Training
import os
import copy
import multiprocessing

def train_model():
    data_dir = 'dataset'
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')

    if not os.path.exists(train_dir) or not os.path.exists(val_dir):
        print("FEHLER: Ordnerstruktur fehlt!")
        return

    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomRotation(degrees=15),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) 
    ])

    val_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
    val_dataset = datasets.ImageFolder(val_dir, transform=val_transforms)

    num_workers = min(4, multiprocessing.cpu_count()) 
    batch_size = 32

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                              num_workers=num_workers, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, 
                            num_workers=num_workers, pin_memory=True, persistent_workers=True)

    print(f"Klassen gefunden: {train_dataset.class_to_idx}")
    print(f"Trainingsbilder: {len(train_dataset)} | Validierungsbilder: {len(val_dataset)}")

    print("\nLade ResNet18...")
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    for param in model.parameters():
        param.requires_grad = False

    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(train_dataset.classes))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Verwende Gerät: {device}")
    if device.type == 'cuda':
        print(f"Grafikkarte: {torch.cuda.get_device_name(0)}")
        
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=0.001)
    
    scaler = GradScaler()

    print("\n=== PHASE 1: Warm-up (Nur letzte Schicht trainieren) ===")
    epochs_phase1 = 5
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(epochs_phase1):
        model.train()
        running_loss, corrects = 0.0, 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            optimizer.zero_grad()
            
            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            corrects += torch.sum(preds == labels.data)
        
        epoch_loss = running_loss / len(train_dataset)
        epoch_acc = corrects.double() / len(train_dataset)

        model.eval()
        val_corrects = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                with autocast():
                    outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                val_corrects += torch.sum(preds == labels.data)
                
        val_acc = val_corrects.double() / len(val_dataset)
        print(f"Epoche {epoch+1}/{epochs_phase1} | Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:.2%} | Val Acc: {val_acc:.2%}")

    print("\n=== PHASE 2: Fine-Tuning (Gesamtes Modell wird aufgeweicht) ===")
    
    for param in model.parameters():
        param.requires_grad = True

    optimizer_ft = optim.Adam(model.parameters(), lr=1e-4)
    
    epochs_phase2 = 10
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer_ft, T_max=epochs_phase2, eta_min=1e-6)

    for epoch in range(epochs_phase2):
        model.train()
        running_loss, corrects = 0.0, 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            optimizer_ft.zero_grad()
            
            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
            scaler.scale(loss).backward()
            scaler.step(optimizer_ft)
            scaler.update()

            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            corrects += torch.sum(preds == labels.data)
        
        scheduler.step() # learning reate for next epoche
        
        epoch_loss = running_loss / len(train_dataset)
        epoch_acc = corrects.double() / len(train_dataset)

        model.eval()
        val_corrects = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                with autocast():
                    outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                val_corrects += torch.sum(preds == labels.data)
                
        val_acc = val_corrects.double() / len(val_dataset)
        
        current_lr = optimizer_ft.param_groups[0]['lr']
        print(f"Epoche {epoch+1}/{epochs_phase2} | LR: {current_lr:.1e} | Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:.2%} | Val Acc: {val_acc:.2%}")

        if val_acc > best_acc:
            best_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(), "knoten_resnet18_finetuned.pth")
    print(f"\nTraining beendet! Bestes Modell mit Val Acc: {best_acc:.2%} als 'knoten_resnet18_finetuned.pth' gespeichert.")

if __name__ == '__main__':
    train_model()