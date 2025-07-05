import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os

def create_visuals_for_framework(df, framework_name):
    """
    Erstellt einen Satz von vier Visualisierungen für ein gegebenes Framework.

    Args:
        df (pd.DataFrame): Das DataFrame mit den Ergebnissen.
        framework_name (str): Der Name des Frameworks (z.B. 'pytorch' oder 'tensorflow').
    """
    print(f"\n--- Starte Visualisierung für {framework_name.upper()} ---")
    
    # --- Ausgabeordner definieren und erstellen ---
    output_folder = f'results_{framework_name}_visuals'
    os.makedirs(output_folder, exist_ok=True)

    # Definiere die korrekte Reihenfolge der Modelle
    model_order = ['1_SimpleCNN', '2_MediumCNN', '3_ComplexCNN', '4_UltraComplexCNN']

    # =============================================================================
    # Abbildung 1: Trainingszeit CPU vs. GPU
    # =============================================================================
    print("Erstelle Abbildung 1: Trainingszeit pro Epoche...")
    df_train_time = df[df['dtype'] == 'float32']
    g1 = sns.relplot(
        data=df_train_time, x='batch_size', y='avg_epoch_time_s', hue='device',
        col='model', kind='line', marker='o', palette={'cpu': '#0d47a1', 'cuda': '#ff6f00'},
        col_order=model_order
    )
    g1.fig.suptitle(f'Trainingszeit pro Epoche ({framework_name.upper()})', fontsize=16)
    g1.set_axis_labels("Batch-Größe", "Zeit pro Epoche (s)")
    g1.set_titles("Modell: {col_name}")
    g1.legend.set_title("Gerät")
    g1.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(output_folder, f'{framework_name}_plot_1_training_time.png'))
    plt.close(g1.fig)

    # =============================================================================
    # Abbildung 2: GPU Speedup-Faktor
    # =============================================================================
    print("Erstelle Abbildung 2: GPU Speedup-Faktor...")
    df_pivot = df[df['dtype'] == 'float32'].pivot_table(
        index=['model', 'batch_size'], columns='device', values='avg_epoch_time_s'
    ).reset_index()
    if 'cpu' in df_pivot.columns and 'cuda' in df_pivot.columns:
        df_pivot['speedup'] = df_pivot['cpu'] / df_pivot['cuda']
        g2 = sns.catplot(
            data=df_pivot, x='batch_size', y='speedup', col='model',
            kind='bar', palette='magma', hue='batch_size',
            col_order=model_order
        )
        g2.fig.suptitle(f'GPU Speedup-Faktor ({framework_name.upper()})', fontsize=16)
        g2.set_axis_labels("Batch-Größe", "Speedup (x-fache Geschwindigkeit)")
        g2.set_titles("Modell: {col_name}")
        g2.legend.set_title("Batch-Größe")
        g2.fig.subplots_adjust(top=0.88)
        plt.savefig(os.path.join(output_folder, f'{framework_name}_plot_2_speedup.png'), bbox_inches='tight')
        plt.close(g2.fig)

    # =============================================================================
    # Abbildung 3: VRAM-Nutzung und Datentypen auf der GPU
    # =============================================================================
    print("Erstelle Abbildung 3: GPU VRAM-Nutzung...")
    df_gpu = df[(df['device'] == 'cuda') & (df['status'] == 'Success')]
    g3 = sns.relplot(
        data=df_gpu, x='batch_size', y='vram_usage_mb', hue='dtype',
        col='model', kind='line', marker='o', 
        palette={'float32': '#c62828', 'float16': '#283593', 'float64': '#2e7d32'},
        col_order=model_order
    )
    g3.fig.suptitle(f'GPU VRAM-Nutzung nach Datentyp ({framework_name.upper()})', fontsize=16)
    g3.set_axis_labels("Batch-Größe", "Speicher (MB)")
    g3.set_titles("Modell: {col_name}")
    g3.legend.set_title("Datentyp")
    g3.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(output_folder, f'{framework_name}_plot_3_vram.png'))
    plt.close(g3.fig)

    # =============================================================================
    # Abbildung 4: Inferenzzeit für den gesamten Testdatensatz
    # =============================================================================
    print("Erstelle Abbildung 4: Inferenzzeit...")
    df_inference = df[(df['dtype'] == 'float32') & (df['status'] == 'Success')]
    g4 = sns.catplot(
        data=df_inference, x='model', y='inference_time_s', hue='device',
        kind='bar', palette={'cpu': '#0d47a1', 'cuda': '#ff6f00'},
        order=model_order, height=6, aspect=1.5
    )
    g4.legend.set_title("Gerät")
    sns.move_legend(g4, "upper left", bbox_to_anchor=(0.9, 0.65))
    g4.fig.suptitle(f'Inferenzzeit für gesamten Testdatensatz ({framework_name.upper()})', fontsize=16)
    g4.set_axis_labels("Modell", "Inferenzzeit (s)")
    g4.set_xticklabels(rotation=15)
    g4.fig.subplots_adjust(top=0.9)
    plt.savefig(os.path.join(output_folder, f'{framework_name}_plot_4_inference_time.png'), bbox_inches='tight')
    plt.close(g4.fig)
    
    print(f"\nAlle Visualisierungen für {framework_name.upper()} wurden im Ordner '{output_folder}' gespeichert.")


if __name__ == '__main__':
    # --- Theme und globale Einstellungen ---
    sns.set_theme(style="whitegrid")
    plt.rcParams['figure.dpi'] = 120

    # Liste der zu verarbeitenden Frameworks und ihrer Ergebnisdateien
    frameworks = {
        'pytorch': 'pytorch_final_results.csv',
        'tensorflow': 'tensorflow_final_results.csv'
    }

    for name, filename in frameworks.items():
        try:
            df_results = pd.read_csv(filename)
            create_visuals_for_framework(df_results, name)
        except FileNotFoundError:
            print(f"\nFEHLER: Die Datei '{filename}' wurde nicht gefunden. Überspringe {name.upper()}.")

    print("\nAlle Aufgaben abgeschlossen.")