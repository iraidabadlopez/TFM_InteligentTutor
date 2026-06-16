from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import cv2
import seaborn as sns
from skimage import io
from sklearn.metrics import confusion_matrix, multilabel_confusion_matrix

random.seed(42)


def show_class_distribution(df):
    plt.figure(figsize=(10, 6))
    sns.countplot(x='label', data=df, palette='viridis')
    plt.title('Distribución de Imágenes por Clase')
    plt.xticks(rotation=45)
    plt.show()

#######################################################################

def show_false_labeling(idx, type, test_paths, num_images=50):

    if num_images > len(idx):
        num_images = len(idx)

    idx = idx[:num_images]
    
    print(f"{type}: {num_images}")

    rows = (num_images - 1) // 5 + 1 if num_images > 0 else 1
    fig = plt.figure(figsize=(15, 3 * rows))

    print(f"=========== {type} ===========")
    for plot_idx, i in enumerate(idx):
        plt.subplot(rows, 5, plot_idx + 1)
        
        img_path = test_paths[i] 

        if not os.path.exists(img_path):
            plt.axis("off")
            plt.title(os.path.basename(img_path), color="red", fontsize=9)
            continue

        img = io.imread(img_path)

        if img.shape[-1] == 1:
            plt.imshow(img.squeeze(), cmap="gray")
        else:
            plt.imshow(img)
            
        plt.axis("off")
        
        # Mostrar sólo el nombre del archivo
        name = os.path.basename(img_path)
        plt.title(name, color="red", fontsize=9, fontweight='bold')

    plt.tight_layout()
    plt.show()

#######################################################################

def show_false_labels_multiclass(test_labels, test_paths, class_names, y_pred_classes, class_ind, type = 'FP'):

    if type == 'FP':
        ind = np.where((y_pred_classes == class_ind) & (test_labels != class_ind))[0]
    elif type == 'FN':
        ind = np.where((y_pred_classes != class_ind) & (test_labels == class_ind))[0]
    else:
        print("Tipo no válido. Use 'FP' para falsos positivos o 'FN' para falsos negativos.")
        return
    
    nombre_clase = class_names[class_ind] # Ej: 'colocarSoporteSup'
    show_false_labeling(
        idx=ind, 
        type=f"Falsos {type} de {nombre_clase}", 
        test_paths=test_paths
    )

#######################################################################
   

def plot_from_history(history):
    hist = history.history if hasattr(history, 'history') else history

    # Determinar cuántas métricas tenemos para ajustar el tamaño de la figura
    has_precision_recall = 'precision' in hist and 'recall' in hist
    rows = 2 if has_precision_recall else 1
    
    plt.figure(figsize=(12, 4 * rows))
    epochs_range = range(len(hist['loss'])) # Usar el largo real de los datos

    # --- 1. Accuracy ---
    plt.subplot(rows, 2, 1)
    plt.plot(epochs_range, hist['accuracy'], label='Train Accuracy')
    plt.plot(epochs_range, hist['val_accuracy'], label='Val Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.legend(loc='lower right')
    plt.grid(True)


    # --- 2. Loss ---
    plt.subplot(rows, 2, 2)
    plt.plot(epochs_range, hist['loss'], label='Train Loss')
    plt.plot(epochs_range, hist['val_loss'], label='Val Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.legend(loc='upper right')
    plt.grid(True)

    # --- 3. Precision y Recall ---
    if has_precision_recall:
        # Precision
        plt.subplot(rows, 2, 3)
        # Nota: Keras a veces nombra la métrica 'precision_1' etc., buscamos la clave
        p_key = [k for k in hist.keys() if 'precision' in k and 'val' not in k][0]
        plt.plot(epochs_range, hist[p_key], label='Train Precision', color='green')
        plt.plot(epochs_range, hist[f'val_{p_key}'], label='Val Precision', color='lightgreen')
        plt.title('Training and Validation Precision')
        plt.xlabel('Epochs')
        plt.legend(loc='lower right')

        # Recall
        plt.subplot(rows, 2, 4)
        r_key = [k for k in hist.keys() if 'recall' in k and 'val' not in k][0]
        plt.plot(epochs_range, hist[r_key], label='Train Recall', color='purple')
        plt.plot(epochs_range, hist[f'val_{r_key}'], label='Val Recall', color='violet')
        plt.title('Training and Validation Recall')
        plt.xlabel('Epochs')
        plt.legend(loc='lower right')

    plt.tight_layout()
    plt.show()

#######################################################################

def show_last_epochs(history, idx=3):
    metrics = history.history if hasattr(history, 'history') else history

    num_epocas_total = len(metrics['loss'])

    epochs_range = range(num_epocas_total - idx + 1, num_epocas_total + 1) 

    train_loss_last = metrics['loss'][-idx:]
    val_loss_last = metrics['val_loss'][-idx:]

    train_acc_last = metrics['accuracy'][-idx:]
    val_acc_last = metrics['val_accuracy'][-idx:]

    plt.figure(figsize=(14, 5))

    # Gráfica de Acc
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, train_acc_last, label='Train Acc', marker='o', linewidth=2, color='green')
    plt.plot(epochs_range, val_acc_last, label='Val Acc', marker='s', linewidth=2, color='orange')
    plt.xticks(epochs_range)
    plt.title(f'Precisión (Accuracy) - Últimas {idx} Épocas')
    plt.xlabel('Época')
    plt.ylabel('Valor')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    # Gráfica de Loss
    plt.subplot(1, 2, 2)
    
    plt.plot(epochs_range, train_loss_last, label='Train Loss', marker='o', linewidth=2)
    plt.plot(epochs_range, val_loss_last, label='Val Loss', marker='s', linewidth=2)
    plt.xticks(epochs_range) # Muestra exactamente los números de las épocas filtradas
    plt.title(f'Pérdida (Loss) - Últimas {idx} Épocas')
    plt.xlabel('Época')
    plt.ylabel('Valor')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.show()

#######################################################################

def show_confusion_matrix(y_true, y_pred, class_names):

    cm = confusion_matrix(y_true, y_pred)

    tn, fp, fn, tp = cm.ravel()

    print("=== MATRIZ DE CONFUSIÓN BINARIA ===")
    print(f"{'Metrica':<20} | {'Valor':<10} | {'Interpretación'}")
    print("-" * 60)

    # Función auxiliar para imprimir filas
    def print_row(label, val, desc):
        print(f"{label:<20} | {val:<10.0f} | {desc}")

    print_row("Verdaderos Neg. (TN)", tn, f"Correctos: {class_names[0]}")
    print_row("Falsos Pos. (FP)", fp, f"Error: Era {class_names[0]} pero predijo {class_names[1]}")
    print_row("Falsos Neg. (FN)", fn, f"Error: Era {class_names[1]} pero predijo {class_names[0]}")
    print_row("Verdaderos Pos. (TP)", tp, f"Correctos: {class_names[1]}")

    print("-" * 60)

    # Resumen de estado
    total_clase_positiva = tp + fn
    umbral = total_clase_positiva * 0.25

    print(f"Estado de la clase {class_names[1]}: ", end="")
    if fn == 0 and fp == 0:
        print("Perfecto")
    elif fn > umbral:
        print(f"Muchos Falsos Negativos: {fn}")
    elif fp > umbral:
        print(f"Muchos Falsos Positivos: {fp}")
    else:
        print("OK")

    return tn, fp, fn, tp

###############################################################################

def show_multiclass_confusion_matrix(test_labels, y_pred_classes, class_names):     
    cm_img = multilabel_confusion_matrix(test_labels, y_pred_classes)

    print("=== MATRICES DE CONFUSIÓN POR CLASE ===")
    print(f"{'Clase':<20} | {'TN':<5} | {'FP':<5} | {'FN':<5} | {'TP':<5} | {'Estado':<15}")
    print("-" * 75)

    for i, matrix in enumerate(cm_img):
        # Desglose de la matriz
        tn = matrix[0, 0]
        fp = matrix[0, 1]
        fn = matrix[1, 0]
        tp = matrix[1, 1]
        
        # Total de imágenes reales de esta clase
        total_clase = tp + fn
        
        umbral_alerta = total_clase * 0.25
        
        nombre_clase = class_names[i]
        
        print(f"{nombre_clase:<20} | {tn:<5.0f} | {fp:<5.0f} | {fn:<5.0f} | {tp:<5.0f} |", end="")
        
        # Lógica de estados personalizada
        if fp == 0 and fn == 0:
            print("Perfecto")
        elif fp > umbral_alerta or fn > umbral_alerta:
            # Alerta si FP o FN superan el 25% de la presencia real de la clase
            print(f" >25%")
        else:
            print("OK")

    print("-" * 75)
    
###############################################################################

def show_imgs(test_paths, y_true, y_pred, class_names, clase_nombre, tipo_caso="FP", max_imagenes=50):
    
    tipo_caso = tipo_caso.upper()
    formatos_validos = ["TP", "TN", "FP", "FN"]
    if tipo_caso not in formatos_validos:
        return

    # 1. Conseguir el ID de la clase
    try:
        id_objetivo = list(class_names).index(clase_nombre)
    except ValueError:
        print(f"La clase '{clase_nombre}' no existe. Opciones: {list(class_names)}")
        return

    indices_filtrados = []

    # 2. Lógica de filtrado según las definiciones matemáticas de la matriz
    for idx in range(len(y_true)):
        real = y_true[idx]
        predicho = y_pred[idx]
        
        if tipo_caso == "TP":
            # Verdadero Positivo: Era la clase y el modelo dijo que era la clase
            condicion = (real == id_objetivo and predicho == id_objetivo)
        elif tipo_caso == "TN":
            # Verdadero Negativo: No era la clase y el modelo dijo otra cosa (no X)
            condicion = (real != id_objetivo and predicho != id_objetivo)
        elif tipo_caso == "FP":
            # Falso Positivo: No era la clase, pero el modelo dijo que SÍ era
            condicion = (real != id_objetivo and predicho == id_objetivo)
        elif tipo_caso == "FN":
            # Falso Negativo: Era la clase, pero el modelo dijo que era OTRA cosa
            condicion = (real == id_objetivo and predicho != id_objetivo)
            
        if condicion:
            indices_filtrados.append(idx)

    print(f"Hay {len(indices_filtrados)} casos de tipo [{tipo_caso}] para la clase '{clase_nombre}'")

    # 3. Renderizar las imágenes encontradas
    if len(indices_filtrados) == 0:
        print(f"No hay imágenes que mostrar para el criterio seleccionado.")
        return

    # Limitar para que el Notebook no colapse si hay cientos de imágenes
    indices_a_mostrar = indices_filtrados[:max_imagenes]
    
    columnas = 3
    filas = (len(indices_a_mostrar) + columnas - 1) // columnas
    plt.figure(figsize=(15, 5 * filas))
    
    for i, idx in enumerate(indices_a_mostrar):
        path_imagen = test_paths[idx]
        clase_real_nombre = class_names[y_true[idx]]
        clase_predicha_nombre = class_names[y_pred[idx]]
        
        # Carga básica con OpenCV cambiando a RGB
        img = cv2.imread(path_imagen)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        plt.subplot(filas, columnas, i + 1)
        plt.imshow(img)
        plt.axis('off')
        
        # El color del título cambia a verde si acertó (TP/TN) o a rojo si falló (FP/FN)
        color_titulo = 'green' if tipo_caso in ["TP", "TN"] else 'red'
        
        plt.title(f"Real: {clase_real_nombre}\nPredicho: {clase_predicha_nombre}\nArchivo: {os.path.basename(path_imagen)}", 
                  color=color_titulo, fontsize=10)
        
    plt.tight_layout()
    plt.show()