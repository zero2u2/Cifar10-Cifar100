"""
Abschlussskript zur Performance-Analyse von CNNs mit TensorFlow

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
import tensorflow as tf
import time
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# =============================================================================
# 1. KONFIGURATION DES EXPERIMENTS
# =============================================================================

# Eine Liste von Batch-Größen, die nacheinander getestet werden sollen.
# Größere Batches können das Training beschleunigen, benötigen aber mehr Speicher.
BATCH_SIZES_TO_TEST = [32, 64, 128, 256]

# Anzahl der Epochen, über die die Trainingszeit gemessen und gemittelt wird.
# Ein Wert > 1 liefert stabilere Zeitmessungen.
NUM_EPOCHS_FOR_TIMING = 3

# Ein Dictionary von Datentypen, die getestet werden sollen.
# 'float32': Standard-Genauigkeit, universell kompatibel.
# 'mixed_float16': Gemischte Genauigkeit. Nutzt float16 für viele Operationen, um die Leistung auf modernen GPUs
#                  (Tensor Cores) erheblich zu steigern und Speicher zu sparen.
# 'float64': Doppelte Genauigkeit. Wird im Deep Learning selten verwendet, da es langsam ist und viel Speicher
#            benötigt, ohne die Genauigkeit bei Bildklassifizierung zu verbessern. Dient hier als Testfall.
DTYPES_TO_TEST = {
    'float32': 'float32',
    'float16': 'mixed_float16',
    'float64': 'float64',
}

# =============================================================================
# 2. MODELLDEFINITIONEN (in TensorFlow/Keras)
# =============================================================================

def SimpleCNN_TF(num_classes=10):
    """
    Stufe 1: Ein sehr einfaches CNN als Baseline.
    - Eine Conv-Schicht zur Merkmalsextraktion.
    - Eine MaxPool-Schicht zur Dimensionsreduktion.
    - Eine Dense-Schicht als Klassifikator.
    """
    return tf.keras.Sequential([
        tf.keras.layers.Input((32, 32, 3)),
        tf.keras.layers.Conv2D(16, 3, padding='same', activation='relu'),
        tf.keras.layers.MaxPool2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(num_classes)
    ])

def MediumCNN_TF(num_classes=10):
    """
    Stufe 2: Ein mittelkomplexes CNN.
    - Zwei separate Conv-Blöcke zur besseren Merkmalsextraktion.
    - Eine kleine Dense-Schicht vor der Ausgabe.
    """
    return tf.keras.Sequential([
        tf.keras.layers.Input((32, 32, 3)),
        tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
        tf.keras.layers.MaxPool2D(),
        tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
        tf.keras.layers.MaxPool2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(num_classes)
    ])

def ComplexCNN_TF(num_classes=10):
    """
    Stufe 3: Ein komplexeres, tieferes CNN.
    - Gestapelte Conv-Schichten innerhalb eines Blocks für komplexere Muster.
    - Eine große Dense-Schicht als leistungsfähiger Klassifikator.
    """
    return tf.keras.Sequential([
        tf.keras.layers.Input((32, 32, 3)),
        tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
        tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
        tf.keras.layers.MaxPool2D(),
        tf.keras.layers.Conv2D(128, 3, padding='same', activation='relu'),
        tf.keras.layers.MaxPool2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(num_classes)
    ])

def UltraComplexCNN_TF(num_classes=10):
    """
    Stufe 4: Ein sehr tiefes und modernes CNN, inspiriert von gängigen Architekturen.
    - Verwendet die Keras Functional API für mehr Flexibilität.
    - Nutzt Blöcke aus (Conv2D -> BatchNormalization -> Mish-Aktivierung).
    - BatchNormalization stabilisiert das Training und beschleunigt die Konvergenz.
    - Mish ist eine moderne, glatte Aktivierungsfunktion, die oft bessere Ergebnisse als ReLU liefert.
    - Dropout wird zur Regularisierung eingesetzt, um Overfitting zu reduzieren.
    """
    inputs = tf.keras.layers.Input(shape=(32, 32, 3))
    
    # Block 1
    x = tf.keras.layers.Conv2D(32, 3, padding='same')(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Mish()(x)
    x = tf.keras.layers.Conv2D(32, 3, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Mish()(x)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    
    # Block 2
    x = tf.keras.layers.Conv2D(64, 3, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Mish()(x)
    x = tf.keras.layers.Conv2D(64, 3, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Mish()(x)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Dropout(0.3)(x)

    # Block 3
    x = tf.keras.layers.Conv2D(128, 3, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Mish()(x)
    x = tf.keras.layers.Conv2D(128, 3, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Mish()(x)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Dropout(0.4)(x)

    # Klassifikator-Teil
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(1024, kernel_regularizer=tf.keras.regularizers.l1_l2(l1=1e-5, l2=1e-4))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Mish()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    
    return tf.keras.Model(inputs=inputs, outputs=outputs)

# Dictionary, das die Namen der Modelle auf die zugehörigen Funktionen abbildet.
# Dies ermöglicht es, einfach durch die zu testenden Modelle zu iterieren.
MODELS_TO_TEST = {
    '1_SimpleCNN': SimpleCNN_TF,
    '2_MediumCNN': MediumCNN_TF,
    '3_ComplexCNN': ComplexCNN_TF,
    '4_UltraComplexCNN': UltraComplexCNN_TF,
}

# =============================================================================
# 3. HELFERFUNKTIONEN
# =============================================================================

def get_data_loaders_tf(batch_size, dtype_policy):
    """
    Lädt den CIFAR-10 Datensatz und bereitet ihn mit der `tf.data` API vor.
    - `.cast()`: Konvertiert die Pixelwerte in den Zieldatentyp.
    - `.shuffle()`: Mischt die Trainingsdaten zufällig.
    - `.batch()`: Fasst Daten zu Batches zusammen.
    - `.prefetch()`: Ermöglicht das Vorladen des nächsten Batches, während der aktuelle verarbeitet wird,
      um die GPU-Auslastung zu maximieren und Engpässe zu vermeiden.
    """
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    x_train = tf.cast(x_train, dtype_policy.compute_dtype) / 255.0
    x_test = tf.cast(x_test, dtype_policy.compute_dtype) / 255.0
    trainloader = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(50000).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    testloader = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return trainloader, testloader

@tf.function
def train_step(model, x, y, optimizer, loss_fn):
    """
    Führt einen einzelnen Trainingsschritt aus.
    Der `@tf.function`-Decorator kompiliert die Funktion in einen performanten TensorFlow-Graphen,
    was die Ausführung erheblich beschleunigt.
    `tf.GradientTape` zeichnet die Operationen auf, um die Gradienten für die Backpropagation zu berechnen.
    """
    with tf.GradientTape() as tape:
        predictions = model(x, training=True)
        loss = loss_fn(y, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

def train_one_epoch_tf(model, loader, optimizer, loss_fn, device_name):
    """Trainiert das Modell für eine komplette Epoche und misst die benötigte Zeit."""
    start_time = time.time()
    with tf.device(device_name):
        for x_batch, y_batch in loader:
            train_step(model, x_batch, y_batch, optimizer, loss_fn)
    end_time = time.time()
    return end_time - start_time

@tf.function
def test_step(model, x):
    """Führt einen reinen Vorwärtsdurchlauf für die Inferenzmessung aus."""
    model(x, training=False)

def measure_inference_time_tf(model, loader, device_name):
    """Misst die Zeit für die Inferenz über den gesamten Testdatensatz."""
    start_time = time.time()
    with tf.device(device_name):
        for x_batch, _ in loader:
            test_step(model, x_batch)
    end_time = time.time()
    return end_time - start_time

def get_vram_usage_mb_tf(device_name):
    """Gibt den maximalen (peak) VRAM-Verbrauch auf der GPU in Megabyte zurück."""
    if 'GPU' in device_name:
        try:
            mem_info = tf.config.experimental.get_memory_info(device_name.replace('/GPU:', 'GPU:'))
            return mem_info['peak'] / (1024 * 1024)
        except: return 0
    return 0

# =============================================================================
# 4. DAS HAUPT-EXPERIMENT
# =============================================================================
if __name__ == '__main__':
    # Geräteerkennung: Prüft, ob GPUs verfügbar sind und fügt sie zur Testliste hinzu.
    # Die Reihenfolge wird hier so festgelegt, dass die GPU zuerst getestet wird.
    gpus = tf.config.list_physical_devices('GPU')
    DEVICES_TO_TEST = {}
    if gpus:
        DEVICES_TO_TEST['/GPU:0'] = 'cuda'
        print(f"CUDA-fähige GPU gefunden: {gpus[0].name}")
    else:
        print("Keine CUDA-fähige GPU gefunden. Tests laufen nur auf der CPU.")
    
    # CPU wird immer zur Testliste hinzugefügt (nach der GPU).
    DEVICES_TO_TEST['/CPU:0'] = 'cpu'

    print("\nStarte Experimente...")
    results = []

    # Hauptschleifen: Iterieren durch alle konfigurierten Kombinationen.
    for tf_device_name, display_device_name in DEVICES_TO_TEST.items():
        for model_name, model_class in MODELS_TO_TEST.items():
            for dtype_name, dtype_policy_str in DTYPES_TO_TEST.items():
                # Überspringen von nicht unterstützten Kombinationen.
                if 'CPU' in tf_device_name and dtype_policy_str == 'mixed_float16':
                    print(f"Info: Überspringe {model_name}, {dtype_name} auf {display_device_name} (nicht unterstützt)")
                    continue

                for batch_size in BATCH_SIZES_TO_TEST:
                    print(f"Teste: {model_name} | {display_device_name} | dtype={dtype_name} | batch_size={batch_size}")
                    try:
                        with tf.device(tf_device_name):
                            # Setzt die globale Policy für den Datentyp (entscheidend für mixed precision).
                            policy = tf.keras.mixed_precision.Policy(dtype_policy_str)
                            tf.keras.mixed_precision.set_global_policy(policy)

                            # Setup: Daten laden, Modell erstellen, Optimizer und Loss definieren.
                            trainloader, testloader = get_data_loaders_tf(batch_size, policy)
                            model = model_class()
                            optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
                            loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

                            # FIX: Initialisiert den Optimierer, indem ein einzelner Trainingsschritt
                            # auf einem Dummy-Batch ausgeführt wird. Dies erzwingt die Erstellung
                            # der Zustandsvariablen des Optimierers vor der @tf.function-Schleife.
                            dummy_x, dummy_y = next(iter(trainloader))
                            with tf.GradientTape() as tape:
                                dummy_predictions = model(dummy_x, training=True)
                                dummy_loss = loss_fn(dummy_y, dummy_predictions)
                            grads = tape.gradient(dummy_loss, model.trainable_variables)
                            optimizer.apply_gradients(zip(grads, model.trainable_variables))


                            # VRAM-Statistik für die GPU zurücksetzen für eine saubere Messung.
                            if 'GPU' in tf_device_name:
                                tf.config.experimental.reset_memory_stats(tf_device_name.replace('/GPU:', 'GPU:'))

                            # Trainingszeit über mehrere Epochen messen und mitteln.
                            epoch_times = [train_one_epoch_tf(model, trainloader, optimizer, loss_fn, tf_device_name) for _ in range(NUM_EPOCHS_FOR_TIMING)]

                            # Metriken nach dem Training sammeln.
                            avg_epoch_time = np.mean(epoch_times)
                            vram_usage = get_vram_usage_mb_tf(tf_device_name)
                            inference_time = measure_inference_time_tf(model, testloader, tf_device_name)

                            # Erfolgreiches Ergebnis speichern.
                            results.append({
                                'model': model_name, 'device': display_device_name, 'dtype': dtype_name,
                                'batch_size': batch_size, 'avg_epoch_time_s': avg_epoch_time,
                                'inference_time_s': inference_time, 'vram_usage_mb': vram_usage, 'status': 'Success'
                            })
                    except (tf.errors.OpError, tf.errors.ResourceExhaustedError, ValueError) as e:
                        # Fehlerbehandlung: Fängt typische Fehler wie "Out of Memory" (OOM) oder nicht unterstützte Operationen ab.
                        error_str = str(e).lower()
                        if 'resourceexhausted' in error_str or 'oom' in error_str:
                            status = 'OutOfMemory'
                        elif 'could not find compiler' in error_str or 'unsupported data type' in error_str:
                            status = 'UnsupportedDType'
                        else:
                            status = 'Error'
                        
                        print(f"  FEHLER ({status}): {e}")
                        
                        # Fehlerhaftes Ergebnis protokollieren.
                        results.append({
                            'model': model_name, 'device': display_device_name, 'dtype': dtype_name,
                            'batch_size': batch_size, 'avg_epoch_time_s': float('nan'),
                            'inference_time_s': float('nan'), 'vram_usage_mb': float('nan'), 'status': status
                        })
                        # Wenn ein Fehler auftritt, macht es keinen Sinn, noch größere Batches zu testen.
                        break
    
    # Nach Abschluss aller Tests werden die Ergebnisse in einem DataFrame zusammengefasst und als CSV gespeichert.
    print("\nAlle Experimente abgeschlossen.")
    df_results = pd.DataFrame(results)
    df_results.to_csv('tensorflow_final_results.csv', index=False)
    print("\nErgebnistabelle:")
    print(df_results)
    print("\nErgebnisse in 'tensorflow_final_results.csv' gespeichert.")

    # =============================================================================
    # 5. ERGEBNISANALYSE UND VISUALISIERUNGS-DASHBOARD
    # =============================================================================
    print("\nErstelle Visualisierungs-Dashboard...")

    # --- Datenvorbereitung für den Speedup-Plot ---
    # Die Daten werden "pivotiert", um CPU- und GPU-Zeiten für dieselbe Konfiguration nebeneinander zu haben.
    df_pivot = df_results.pivot_table(
        index=['model', 'dtype', 'batch_size'], columns='device', values='avg_epoch_time_s'
    ).reset_index()
    if 'cpu' in df_pivot.columns and 'cuda' in df_pivot.columns:
        df_pivot['speedup'] = df_pivot['cpu'] / df_pivot['cuda']
    else:
        df_pivot['speedup'] = np.nan # Speedup kann nicht berechnet werden

    # --- Dashboard erstellen ---
    sns.set_theme(style="whitegrid", palette="viridis")
    fig, axes = plt.subplots(2, 2, figsize=(20, 15))
    fig.suptitle('TensorFlow Performance-Dashboard: CPU vs. GPU für CNN-Training', fontsize=22)

    # --- Plot A: Trainingszeit pro Epoche ---
    # Zeigt, wie sich die Trainingszeit mit der Batch-Größe für verschiedene Geräte und Modelle ändert.
    sns.lineplot(
        ax=axes[0, 0], data=df_results, x='batch_size', y='avg_epoch_time_s',
        hue='device', style='model', marker='o', palette='viridis'
    ).set_title('A) Trainingszeit pro Epoche', fontsize=16)
    axes[0, 0].set_xlabel('Batch-Größe', fontsize=12)
    axes[0, 0].set_ylabel('Zeit (Sekunden)', fontsize=12)

    # --- Plot B: GPU Speedup-Faktor ---
    # Zeigt, um wie viel Mal schneller die GPU im Vergleich zur CPU ist.
    if not df_pivot['speedup'].isnull().all():
        sns.barplot(
            ax=axes[0, 1], data=df_pivot, x='batch_size', y='speedup',
            hue='model', palette='magma'
        ).set_title('B) GPU Speedup-Faktor (x-fache Geschwindigkeit)', fontsize=16)
        axes[0, 1].axhline(1, ls='--', color='black') # Linie bei 1x Speedup
        axes[0, 1].set_xlabel('Batch-Größe', fontsize=12)
        axes[0, 1].set_ylabel('Speedup (CPU-Zeit / GPU-Zeit)', fontsize=12)

    # --- Plot C: GPU VRAM-Nutzung ---
    # Zeigt den Speicherverbrauch auf der GPU in Abhängigkeit von Batch-Größe, Modell und Datentyp.
    df_gpu = df_results[(df_results['device'] == 'cuda') & (df_results['status'] == 'Success')]
    if not df_gpu.empty:
        sns.lineplot(
            ax=axes[1, 0], data=df_gpu, x='batch_size', y='vram_usage_mb',
            hue='model', style='dtype', marker='o', palette='crest'
        ).set_title('C) GPU VRAM-Nutzung', fontsize=16)
    axes[1, 0].set_xlabel('Batch-Größe', fontsize=12)
    axes[1, 0].set_ylabel('Speicher (MB)', fontsize=12)

    # --- Plot D: Inferenzzeit ---
    # Vergleicht die Zeit für die Klassifizierung des gesamten Testdatensatzes.
    sns.barplot(
        ax=axes[1, 1], data=df_results[df_results['status']=='Success'],
        x='device', y='inference_time_s', hue='model', palette='rocket'
    ).set_title('D) Inferenzzeit für gesamten Testdatensatz', fontsize=16)
    axes[1, 1].set_xlabel('Gerät', fontsize=12)
    axes[1, 1].set_ylabel('Zeit (Sekunden)', fontsize=12)
    
    # Layout anpassen und das Dashboard als Bild speichern.
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.savefig('tensorflow_final_dashboard.png', dpi=150)
    plt.show()

    print("\nDashboard als 'tensorflow_final_dashboard.png' gespeichert.")