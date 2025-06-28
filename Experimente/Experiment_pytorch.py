"""
Abschlussskript zur Performance-Analyse von CNNs mit PyTorch

Dieses Skript führt eine systematische Analyse der Trainings- und Inferenz-Performance
verschiedener Convolutional Neural Network (CNN) Architekturen durch. Es vergleicht
die Leistung auf CPU und GPU unter Verwendung unterschiedlicher numerischer
Genauigkeiten (float32, float16, float64) und variierender Batch-Größen.

Das Experiment ist wie folgt aufgebaut:
1.  **Konfiguration**: Definition der zu testenden Parameter (Batch-Größen, Datentypen).
2.  **Modell-Definitionen**: Implementierung von vier CNNs mit steigender Komplexität.
3.  **Helferfunktionen**: Kapselung von Logik für Datenladen, Training und Messung.
4.  **Experiment-Durchführung**: Systematisches Testen aller Konfigurationen und Sammeln der Ergebnisse.
5.  **Visualisierung**: Erstellung eines Dashboards mit vier Plots zur Analyse der Ergebnisse
    (Trainingszeit, Speedup, VRAM-Nutzung, Inferenzzeit).

Die Ergebnisse werden in einer CSV-Datei und als PNG-Dashboard gespeichert.
"""

# Importieren der notwendigen Bibliotheken
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import time
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# =============================================================================
# 1. KONFIGURATION DES EXPERIMENTS
# =============================================================================

# Eine Liste von Batch-Größen, die nacheinander getestet werden sollen.
BATCH_SIZES_TO_TEST = [32, 64, 128, 256]

# Anzahl der Epochen, über die die Trainingszeit gemessen und gemittelt wird.
NUM_EPOCHS_FOR_TIMING = 3

# Ein Dictionary von Datentypen, die getestet werden sollen.
DTYPES_TO_TEST = {
    'float32': torch.float32,
    'float16': torch.float16,
    'float64': torch.float64,
}

# =============================================================================
# 2. MODELLDEFINITIONEN (4 STUFEN)
# =============================================================================

# Stufe 1: Einfachstes Modell
class SimpleCNN(nn.Module):
    """
    Stufe 1: Ein sehr einfaches CNN als Baseline.
    - Eine Conv-Schicht zur Merkmalsextraktion.
    - Eine MaxPool-Schicht zur Dimensionsreduktion.
    - Eine Dense-Schicht als Klassifikator.
    """
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(nn.Conv2d(3, 16, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2, 2))
        self.classifier = nn.Linear(16 * 16 * 16, num_classes)
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1) # Flatten-Operation
        return self.classifier(x)

# Stufe 2: Modell des Dozenten (aus Notebook 05)
class MediumCNN(nn.Module):
    """
    Stufe 2: Ein mittelkomplexes CNN.
    - Zwei separate Conv-Blöcke zur besseren Merkmalsextraktion.
    - Eine kleine Dense-Schicht vor der Ausgabe.
    """
    def __init__(self, num_classes=10):
        super(MediumCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2, 2)
        )
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(64*8*8, 64), nn.ReLU(), nn.Linear(64, num_classes))
    def forward(self, x):
        return self.classifier(self.features(x))

# Stufe 3: Von uns erweitertes, komplexes Modell
class ComplexCNN(nn.Module):
    """
    Stufe 3: Ein komplexeres, tieferes CNN.
    - Gestapelte Conv-Schichten innerhalb eines Blocks für komplexere Muster.
    - Eine große Dense-Schicht als leistungsfähiger Klassifikator.
    """
    def __init__(self, num_classes=10):
        super(ComplexCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, 1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2, 2)
        )
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(128*8*8, 512), nn.ReLU(), nn.Linear(512, num_classes))
    def forward(self, x):
        return self.classifier(self.features(x))

# Stufe 4: Modell des Kommilitonen
class UltraComplexCNN(nn.Module):
    """
    Stufe 4: Ein sehr tiefes und modernes CNN, inspiriert von gängigen Architekturen.
    - Nutzt Blöcke aus (Conv2D -> BatchNormalization -> Mish-Aktivierung).
    - BatchNormalization stabilisiert das Training und beschleunigt die Konvergenz.
    - Mish ist eine moderne, glatte Aktivierungsfunktion, die oft bessere Ergebnisse als ReLU liefert.
    - Dropout wird zur Regularisierung eingesetzt, um Overfitting zu reduzieren.
    """
    def __init__(self, num_classes=10):
        super(UltraComplexCNN, self).__init__()
        # Jeder Block besteht aus zwei Conv-BN-Mish-Schichten, gefolgt von Max-Pooling und Dropout.
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.Mish(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.Mish(),
            nn.MaxPool2d(2), nn.Dropout(0.2)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.Mish(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.Mish(),
            nn.MaxPool2d(2), nn.Dropout(0.3)
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.Mish(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.Mish(),
            nn.MaxPool2d(2), nn.Dropout(0.4)
        )
        self.block4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.Mish(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.Mish(),
            nn.MaxPool2d(2), nn.Dropout(0.5)
        )
        # Der Klassifikator-Teil verwendet ebenfalls BN und Dropout zwischen den Dense-Layern.
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 2 * 2, 1024), nn.BatchNorm1d(1024), nn.Mish(), nn.Dropout(0.5),
            nn.Linear(1024, 256), nn.BatchNorm1d(256), nn.Mish(), nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        return self.classifier(x)

# Dictionary, das die Namen der Modelle auf die zugehörigen Klassen abbildet.
MODELS_TO_TEST = {
    '1_SimpleCNN': SimpleCNN,
    '2_MediumCNN': MediumCNN,
    '3_ComplexCNN': ComplexCNN,
    '4_UltraComplexCNN': UltraComplexCNN,
}

# =============================================================================
# 3. HELFERFUNKTIONEN
# =============================================================================
def get_data_loaders(batch_size, data_root='./data'):
    """
    Lädt den CIFAR-10 Datensatz und erstellt die DataLoader.
    - `transforms.ToTensor()`: Konvertiert Bilder in PyTorch-Tensoren und skaliert Pixel auf [0, 1].
    - `num_workers=2`: Nutzt 2 parallele Prozesse zum Laden der Daten, um die CPU-GPU-Pipeline zu beschleunigen.
    - `pin_memory=True`: Fixiert die Daten im Speicher, was den Transfer zur GPU beschleunigt.
    """
    transform = transforms.Compose([transforms.ToTensor()])
    trainset = torchvision.datasets.CIFAR10(root=data_root, train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    testset = torchvision.datasets.CIFAR10(root=data_root, train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    return trainloader, testloader

def train_one_epoch(model, loader, optimizer, loss_fn, device):
    """Trainiert das Modell für eine komplette Epoche und misst die benötigte Zeit."""
    model.train() # Setzt das Modell in den Trainingsmodus (aktiviert z.B. Dropout).
    start_time = time.time()
    # Holt den Datentyp direkt vom ersten Parameter des Modells. Dies ist robust und funktioniert für jede Architektur.
    model_dtype = next(model.parameters()).dtype
    for data, target in loader:
        data, target = data.to(device, dtype=model_dtype), target.to(device)
        optimizer.zero_grad() # Setzt die Gradienten aus dem letzten Schritt zurück.
        output = model(data)
        loss = loss_fn(output, target)
        loss.backward() # Berechnet die Gradienten (Backpropagation).
        optimizer.step() # Aktualisiert die Gewichte des Modells.
    if device.type == 'cuda':
        torch.cuda.synchronize() # Wartet, bis alle GPU-Operationen abgeschlossen sind, für eine genaue Zeitmessung.
    end_time = time.time()
    return end_time - start_time

def measure_inference_time(model, loader, device):
    """Misst die Zeit für die Inferenz über den gesamten Testdatensatz."""
    model.eval() # Setzt das Modell in den Evaluationsmodus (deaktiviert z.B. Dropout).
    start_time = time.time()
    model_dtype = next(model.parameters()).dtype
    with torch.no_grad(): # Deaktiviert die Gradientenberechnung, um Speicher und Rechenzeit zu sparen.
        for data, _ in loader:
            data = data.to(device, dtype=model_dtype)
            model(data)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    end_time = time.time()
    return end_time - start_time

def get_vram_usage_mb(device):
    """Gibt den maximalen (peak) VRAM-Verbrauch auf der GPU in Megabyte zurück."""
    if device.type == 'cuda':
        return torch.cuda.max_memory_allocated(device) / (1024 * 1024)
    return 0

# =============================================================================
# 4. DAS HAUPT-EXPERIMENT
# =============================================================================
if __name__ == '__main__':
    # Geräteerkennung
    DEVICES_TO_TEST = {'cpu': 'cpu'}
    if torch.cuda.is_available():
        DEVICES_TO_TEST['cuda'] = 'cuda'
        print(f"CUDA-fähige GPU gefunden: {torch.cuda.get_device_name(0)}")
    else:
        print("Keine CUDA-fähige GPU gefunden. Tests laufen nur auf der CPU.")

    print("\nStarte Experimente...")
    results = []

    # Hauptschleifen: Iterieren durch alle konfigurierten Kombinationen.
    for torch_device_name, display_device_name in DEVICES_TO_TEST.items():
        device = torch.device(torch_device_name)
        for model_name, model_class in MODELS_TO_TEST.items():
            for dtype_name, dtype in DTYPES_TO_TEST.items():
                # Überspringen von nicht unterstützten Kombinationen.
                if device.type == 'cpu' and dtype == torch.float16:
                    print(f"Info: Überspringe {model_name}, {dtype_name} auf {display_device_name} (nicht unterstützt)")
                    continue

                for batch_size in BATCH_SIZES_TO_TEST:
                    print(f"Teste: {model_name} | {display_device_name} | dtype={dtype_name} | batch_size={batch_size}")
                    try:
                        # Setup
                        trainloader, testloader = get_data_loaders(batch_size)
                        model = model_class().to(device=device, dtype=dtype)
                        optimizer = optim.SGD(model.parameters(), lr=0.01)
                        loss_fn = nn.CrossEntropyLoss()
                        
                        if device.type == 'cuda':
                            torch.cuda.reset_peak_memory_stats(device)

                        # Messung
                        epoch_times = [train_one_epoch(model, trainloader, optimizer, loss_fn, device) for _ in range(NUM_EPOCHS_FOR_TIMING)]
                        avg_epoch_time = np.mean(epoch_times)
                        vram_usage = get_vram_usage_mb(device)
                        inference_time = measure_inference_time(model, testloader, device)

                        # Erfolgreiches Ergebnis speichern
                        results.append({
                            'model': model_name, 'device': display_device_name, 'dtype': dtype_name,
                            'batch_size': batch_size, 'avg_epoch_time_s': avg_epoch_time,
                            'inference_time_s': inference_time, 'vram_usage_mb': vram_usage, 'status': 'Success'
                        })
                    except RuntimeError as e:
                        # Robuste Fehlerbehandlung
                        error_str = str(e).lower()
                        if "not implemented for 'double'" in error_str: status = 'UnsupportedDType'
                        elif 'out of memory' in error_str: status = 'OutOfMemory'
                        else: status = 'Error'
                        
                        print(f"  FEHLER ({status}): {e}")
                        results.append({
                            'model': model_name, 'device': display_device_name, 'dtype': dtype_name,
                            'batch_size': batch_size, 'avg_epoch_time_s': float('nan'),
                            'inference_time_s': float('nan'), 'vram_usage_mb': float('nan'), 'status': status
                        })
                        # Breche die Batch-Schleife für diese Konfiguration ab, da sie fehlschlägt.
                        break
    
    print("\nAlle Experimente abgeschlossen.")
    df_results = pd.DataFrame(results)
    df_results.to_csv('pytorch_final_results.csv', index=False)
    print("\nErgebnistabelle:")
    print(df_results)
    print("\nErgebnisse in 'pytorch_final_results.csv' gespeichert.")
    
    # =============================================================================
    # 5. ERGEBNISANALYSE UND VISUALISIERUNGS-DASHBOARD
    # =============================================================================
    print("\nErstelle Visualisierungs-Dashboard...")

    # --- Datenvorbereitung für den Speedup-Plot ---
    df_pivot = df_results.pivot_table(index=['model', 'dtype', 'batch_size'], columns='device', values='avg_epoch_time_s').reset_index()
    if 'cpu' in df_pivot.columns and 'cuda' in df_pivot.columns:
        df_pivot['speedup'] = df_pivot['cpu'] / df_pivot['cuda']
    else:
        df_pivot['speedup'] = np.nan

    # --- Dashboard erstellen ---
    sns.set_theme(style="whitegrid", palette="viridis")
    fig, axes = plt.subplots(2, 2, figsize=(20, 15))
    fig.suptitle('PyTorch Performance-Dashboard: CPU vs. GPU für CNN-Training', fontsize=22)

    # --- Plot A: Trainingszeit ---
    sns.lineplot(ax=axes[0, 0], data=df_results, x='batch_size', y='avg_epoch_time_s', hue='device', style='model', marker='o', palette='viridis').set_title('A) Trainingszeit pro Epoche', fontsize=16)
    axes[0, 0].set_xlabel('Batch-Größe', fontsize=12)
    axes[0, 0].set_ylabel('Zeit (Sekunden)', fontsize=12)

    # --- Plot B: Speedup-Faktor ---
    if not df_pivot['speedup'].isnull().all():
        sns.barplot(ax=axes[0, 1], data=df_pivot, x='batch_size', y='speedup', hue='model', palette='magma').set_title('B) GPU Speedup-Faktor (x-fache Geschwindigkeit)', fontsize=16)
        axes[0, 1].axhline(1, ls='--', color='black')
        axes[0, 1].set_xlabel('Batch-Größe', fontsize=12)
        axes[0, 1].set_ylabel('Speedup (CPU-Zeit / GPU-Zeit)', fontsize=12)

    # --- Plot C: VRAM-Nutzung ---
    df_gpu = df_results[(df_results['device'] == 'cuda') & (df_results['status'] == 'Success')]
    if not df_gpu.empty:
        sns.lineplot(ax=axes[1, 0], data=df_gpu, x='batch_size', y='vram_usage_mb', hue='model', style='dtype', marker='o', palette='crest').set_title('C) GPU VRAM-Nutzung', fontsize=16)
    axes[1, 0].set_xlabel('Batch-Größe', fontsize=12)
    axes[1, 0].set_ylabel('Speicher (MB)', fontsize=12)

    # --- Plot D: Inferenzzeit ---
    sns.barplot(ax=axes[1, 1], data=df_results[df_results['status']=='Success'], x='device', y='inference_time_s', hue='model', palette='rocket').set_title('D) Inferenzzeit für gesamten Testdatensatz', fontsize=16)
    axes[1, 1].set_xlabel('Gerät', fontsize=12)
    axes[1, 1].set_ylabel('Zeit (Sekunden)', fontsize=12)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.savefig('pytorch_final_dashboard.png', dpi=150)
    plt.show()

    print("\nDashboard als 'pytorch_final_dashboard.png' gespeichert.")