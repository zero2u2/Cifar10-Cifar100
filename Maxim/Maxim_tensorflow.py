# Finaler TensorFlow-Code: Analyse mit 3 Komplexitätsstufen und 3 Datentypen

import tensorflow as tf
import time
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# =============================================================================
# 1. KONFIGURATION
# =============================================================================
BATCH_SIZES_TO_TEST = [32, 64, 128, 256]
NUM_EPOCHS_FOR_TIMING = 3

# *** HIER IST DIE ANPASSUNG: float64 wurde hinzugefügt ***
DTYPES_TO_TEST = {
    'float32': 'float32',
    'float16': 'mixed_float16',
    'float64': 'float64',
}

# =============================================================================
# 2. MODELLDEFINITIONEN (3 STUFEN)
# =============================================================================

# Stufe 1: Einfachstes Modell
def SimpleCNN_TF(num_classes=10):
    return tf.keras.Sequential([
        tf.keras.layers.Input(shape=(32, 32, 3)),
        tf.keras.layers.Conv2D(16, kernel_size=3, padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=2, strides=2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(num_classes)
    ])

# Stufe 2: Mittleres Modell von Oli
def MediumCNN_TF(num_classes=10):
    return tf.keras.Sequential([
        tf.keras.layers.Input(shape=(32, 32, 3)),
        tf.keras.layers.Conv2D(32, kernel_size=3, padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=2, strides=2),
        tf.keras.layers.Conv2D(64, kernel_size=3, padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=2, strides=2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(num_classes)
    ])

# Stufe 3: komplexeres Modell
def ComplexCNN_TF(num_classes=10):
    inputs = tf.keras.Input(shape=(32, 32, 3))
    x = tf.keras.layers.Conv2D(32, kernel_size=3, padding='same', activation='relu')(inputs)
    x = tf.keras.layers.Conv2D(64, kernel_size=3, padding='same', activation='relu')(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=2, strides=2)(x)
    x = tf.keras.layers.Conv2D(128, kernel_size=3, padding='same', activation='relu')(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=2, strides=2)(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(512, activation='relu')(x)
    outputs = tf.keras.layers.Dense(num_classes)(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs)


MODELS_TO_TEST = {
    '1_SimpleCNN': SimpleCNN_TF,
    '2_MediumCNN_Dozent': MediumCNN_TF,
    '3_ComplexCNN': ComplexCNN_TF
}

# =============================================================================
# 3. HELFERFUNKTIONEN
# =============================================================================
def get_data_loaders_tf(batch_size, dtype_policy):
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    x_train = tf.cast(x_train, dtype_policy.compute_dtype) / 255.0
    x_test = tf.cast(x_test, dtype_policy.compute_dtype) / 255.0
    trainloader = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(50000).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    testloader = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return trainloader, testloader

@tf.function
def train_step(model, x, y, optimizer, loss_fn):
    with tf.GradientTape() as tape:
        predictions = model(x, training=True)
        loss = loss_fn(y, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

def train_one_epoch_tf(model, loader, optimizer, loss_fn, device_name):
    start_time = time.time()
    with tf.device(device_name):
        for x_batch, y_batch in loader:
            train_step(model, x_batch, y_batch, optimizer, loss_fn)
    end_time = time.time()
    return end_time - start_time

@tf.function
def test_step(model, x):
    model(x, training=False)

def measure_inference_time_tf(model, loader, device_name):
    start_time = time.time()
    with tf.device(device_name):
        for x_batch, _ in loader:
            test_step(model, x_batch)
    end_time = time.time()
    return end_time - start_time

def get_vram_usage_mb_tf(device_name):
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
    gpus = tf.config.list_physical_devices('GPU')
    DEVICES_TO_TEST = {'/CPU:0': 'cpu'}
    if gpus:
        DEVICES_TO_TEST['/GPU:0'] = 'cuda'
        print(f"CUDA-fähige GPU gefunden: {gpus[0].name}")
    else:
        print("Keine CUDA-fähige GPU gefunden. Tests laufen nur auf der CPU.")

    print("\nStarte Experimente...")
    results = []

    for tf_device_name, display_device_name in DEVICES_TO_TEST.items():
        for model_name, model_class in MODELS_TO_TEST.items():
            for dtype_name, dtype_policy_str in DTYPES_TO_TEST.items():
                if 'CPU' in tf_device_name and dtype_policy_str == 'mixed_float16':
                    print(f"Info: Überspringe {model_name}, {dtype_name} auf {display_device_name} (nicht unterstützt)")
                    continue

                for batch_size in BATCH_SIZES_TO_TEST:
                    print(f"Teste: {model_name} | {display_device_name} | dtype={dtype_name} | batch_size={batch_size}")
                    try:
                        with tf.device(tf_device_name):
                            policy = tf.keras.mixed_precision.Policy(dtype_policy_str)
                            tf.keras.mixed_precision.set_global_policy(policy)

                            trainloader, testloader = get_data_loaders_tf(batch_size, policy)
                            model = model_class()
                            optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
                            loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

                            if 'GPU' in tf_device_name:
                                tf.config.experimental.reset_memory_stats(tf_device_name.replace('/GPU:', 'GPU:'))

                            epoch_times = [train_one_epoch_tf(model, trainloader, optimizer, loss_fn, tf_device_name) for _ in range(NUM_EPOCHS_FOR_TIMING)]

                            avg_epoch_time = np.mean(epoch_times)
                            vram_usage = get_vram_usage_mb_tf(tf_device_name)
                            inference_time = measure_inference_time_tf(model, testloader, tf_device_name)

                            results.append({
                                'model': model_name, 'device': display_device_name, 'dtype': dtype_name,
                                'batch_size': batch_size, 'avg_epoch_time_s': avg_epoch_time,
                                'inference_time_s': inference_time, 'vram_usage_mb': vram_usage, 'status': 'Success'
                            })
                    except (tf.errors.OpError, tf.errors.ResourceExhaustedError, ValueError) as e:
                        error_str = str(e).lower()
                        if 'resourceexhausted' in error_str or 'oom' in error_str:
                            status = 'OutOfMemory'
                        elif 'could not find compiler' in error_str or 'unsupported data type' in error_str:
                            status = 'UnsupportedDType'
                        else:
                            status = 'Error'
                        
                        print(f"  FEHLER ({status}): {e}")
                        
                        results.append({
                            'model': model_name, 'device': display_device_name, 'dtype': dtype_name,
                            'batch_size': batch_size, 'avg_epoch_time_s': float('nan'),
                            'inference_time_s': float('nan'), 'vram_usage_mb': float('nan'), 'status': status
                        })
                        break
    
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

    df_pivot = df_results.pivot_table(
        index=['model', 'dtype', 'batch_size'], columns='device', values='avg_epoch_time_s'
    ).reset_index()
    if 'cpu' in df_pivot.columns and 'cuda' in df_pivot.columns:
        df_pivot['speedup'] = df_pivot['cpu'] / df_pivot['cuda']
    else:
        df_pivot['speedup'] = np.nan

    sns.set_theme(style="whitegrid", palette="viridis")
    fig, axes = plt.subplots(2, 2, figsize=(20, 15))
    fig.suptitle('TensorFlow Performance-Dashboard: CPU vs. GPU für CNN-Training', fontsize=22)

    sns.lineplot(
        ax=axes[0, 0], data=df_results, x='batch_size', y='avg_epoch_time_s',
        hue='device', style='model', marker='o', palette='viridis'
    ).set_title('A) Trainingszeit pro Epoche', fontsize=16)
    axes[0, 0].set_xlabel('Batch-Größe', fontsize=12)
    axes[0, 0].set_ylabel('Zeit (Sekunden)', fontsize=12)

    if not df_pivot['speedup'].isnull().all():
        sns.barplot(
            ax=axes[0, 1], data=df_pivot, x='batch_size', y='speedup',
            hue='model', palette='magma'
        ).set_title('B) GPU Speedup-Faktor (x-fache Geschwindigkeit)', fontsize=16)
        axes[0, 1].axhline(1, ls='--', color='black')
        axes[0, 1].set_xlabel('Batch-Größe', fontsize=12)
        axes[0, 1].set_ylabel('Speedup (CPU-Zeit / GPU-Zeit)', fontsize=12)

    df_gpu = df_results[(df_results['device'] == 'cuda') & (df_results['status'] == 'Success')]
    if not df_gpu.empty:
        sns.lineplot(
            ax=axes[1, 0], data=df_gpu, x='batch_size', y='vram_usage_mb',
            hue='model', style='dtype', marker='o', palette='crest'
        ).set_title('C) GPU VRAM-Nutzung', fontsize=16)
    axes[1, 0].set_xlabel('Batch-Größe', fontsize=12)
    axes[1, 0].set_ylabel('Speicher (MB)', fontsize=12)

    sns.barplot(
        ax=axes[1, 1], data=df_results[df_results['status']=='Success'],
        x='device', y='inference_time_s', hue='model', palette='rocket'
    ).set_title('D) Inferenzzeit für gesamten Testdatensatz', fontsize=16)
    axes[1, 1].set_xlabel('Gerät', fontsize=12)
    axes[1, 1].set_ylabel('Zeit (Sekunden)', fontsize=12)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.savefig('tensorflow_final_dashboard.png', dpi=150)
    plt.show()

    print("\nDashboard als 'tensorflow_final_dashboard.png' gespeichert.")