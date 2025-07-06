import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt # Import für die Visualisierung
import pandas as pd # NEU: Import für CSV-Export

# =============================================================================
# 1. KONFIGURATION
# =============================================================================
N_SPLITS = 10
N_EPOCHS = 10
BATCH_SIZE = 64
LEARNING_RATE = 0.001 
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# =============================================================================
# 2. MODELLDEFINITION
# =============================================================================
class UltraComplexCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(UltraComplexCNN, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout(0.2)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout(0.3)
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout(0.4)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 512), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    def forward(self, x):
        x = self.block1(x); x = self.block2(x); x = self.block3(x)
        return self.classifier(x)

# =============================================================================
# 3. HELFERFUNKTIONEN
# =============================================================================
def train_model(model, train_loader, optimizer, loss_fn, device):
    model.train()
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()

def evaluate_model(model, val_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    return 100 * correct / total

# =============================================================================
# 4. DAS KREUZVALIDIERUNGS-EXPERIMENT
# =============================================================================
if __name__ == '__main__':
    print(f"Starte {N_SPLITS}-fache Kreuzvalidierung auf Gerät: {DEVICE}")
    device = torch.device(DEVICE)

    transform = transforms.Compose([transforms.ToTensor()])
    full_train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    
    X = full_train_dataset.data
    y = np.array(full_train_dataset.targets)
    
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    
    # *** ANPASSUNG: Verlauf der Genauigkeiten speichern ***
    history = {}

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"\n--- Starte Fold {fold+1}/{N_SPLITS} ---")

        train_subset = torch.utils.data.Subset(full_train_dataset, train_idx)
        val_subset = torch.utils.data.Subset(full_train_dataset, val_idx)
        
        train_loader = torch.utils.data.DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False)
        
        model = UltraComplexCNN().to(device)
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        loss_fn = nn.CrossEntropyLoss()

        # *** ANPASSUNG: Liste für den Verlauf dieses Folds ***
        fold_history = []
        for epoch in range(N_EPOCHS):
            train_model(model, train_loader, optimizer, loss_fn, device)
            current_acc = evaluate_model(model, val_loader, device)
            fold_history.append(current_acc) # Genauigkeit dieser Epoche speichern
            print(f"  Epoch {epoch+1}/{N_EPOCHS}, Validierungs-Genauigkeit: {current_acc:.2f}%")
        
        history[f'Fold {fold+1}'] = fold_history

    # Endergebnis berechnen
    final_accuracies = [hist[-1] for hist in history.values()]
    mean_accuracy = np.mean(final_accuracies)
    std_accuracy = np.std(final_accuracies)

    print("\n=======================================================")
    print(f"Ergebnis der {N_SPLITS}-fachen Kreuzvalidierung:")
    print(f"Mittlere Genauigkeit: {mean_accuracy:.2f}%")
    print(f"Standardabweichung der Genauigkeit: {std_accuracy:.2f}")
    print("=======================================================")

    # =============================================================================
    # *** NEU: 5. VISUALISIERUNG DER ERGEBNISSE ***
    # =============================================================================
    print("\nErstelle Visualisierung der Kreuzvalidierungs-Ergebnisse...")
    
    # Erstelle eine Figur mit zwei Subplots untereinander
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 14))
    fig.suptitle('Ergebnisse der 10-fachen Kreuzvalidierung', fontsize=18)

    # --- Plot 1: Balkendiagramm der finalen Genauigkeiten ---
    ax1.bar(history.keys(), final_accuracies, color=sns.color_palette('viridis', n_colors=N_SPLITS))
    ax1.axhline(mean_accuracy, color='r', linestyle='--', label=f'Mittelwert: {mean_accuracy:.2f}%')
    ax1.set_title('Finale Genauigkeit pro Fold', fontsize=14)
    ax1.set_ylabel('Genauigkeit (%)')
    ax1.set_ylim(bottom=max(0, min(final_accuracies) - 5)) # Achsen-Limit anpassen
    ax1.legend()

    # --- Plot 2: Liniendiagramm der Lernkurven ---
    for fold_name, accuracies in history.items():
        ax2.plot(range(1, N_EPOCHS + 1), accuracies, marker='o', linestyle='-', label=fold_name)
    ax2.set_title('Lernkurven der Validierungs-Genauigkeit pro Fold', fontsize=14)
    ax2.set_xlabel('Epoche')
    ax2.set_ylabel('Genauigkeit (%)')
    ax2.set_xticks(range(1, N_EPOCHS + 1))
    ax2.legend()
    ax2.grid(True, which='both', linestyle='--', linewidth=0.5)

    # Layout anpassen und speichern
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig('cross_validation_results.png', dpi=120)
    plt.show()

    print("\nVisualisierung als 'cross_validation_results.png' gespeichert.")

    # =============================================================================
    # *** NEU: 6. ERGEBNISSE IN CSV SPEICHERN ***
    # =============================================================================
    print("\nSpeichere detaillierte Ergebnisse in CSV-Datei...")
    
    # DataFrame aus dem 'history'-Dictionary erstellen
    results_df = pd.DataFrame(history)
    
    # Den Index anpassen, sodass er bei 1 beginnt (für die Epochen)
    results_df.index = np.arange(1, len(results_df) + 1)
    results_df.index.name = 'Epoch'
    
    # DataFrame in eine CSV-Datei speichern
    csv_filename = 'cross_validation_history.csv'
    results_df.to_csv(csv_filename)
    
    print(f"Ergebnisse als '{csv_filename}' gespeichert.")