"""Módulo con utilidades para visualización y manejo de datos."""

import os
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

# Configurar estilo de las gráficas
plt.style.use("ggplot")
sns.set_theme(style="whitegrid")
plt.rcParams["figure.facecolor"] = "white"


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    classes: List[str],
    save_path: Optional[str] = None,
    show_plot: bool = True,
    normalize: bool = True,
    title: str = "Matriz de confusión",
    figsize: Tuple[int, int] = (10, 8),
) -> np.ndarray:
    """
    Calcula y grafica la matriz de confusión.

    Args:
        y_true: Etiquetas verdaderas.
        y_pred: Predicciones del modelo.
        classes: Lista con los nombres de las clases.
        save_path: Ruta para guardar la figura. Si es None, no se guarda.
        show_plot: Si es True, muestra la figura.
        normalize: Si es True, normaliza la matriz por filas.
        title: Título del gráfico.
        figsize: Tamaño de la figura (ancho, alto).

    Returns:
        Matriz de confusión.
    """
    # Calcular matriz de confusión
    cm = confusion_matrix(y_true, y_pred)

    # Normalizar si es necesario
    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        cm = np.nan_to_num(cm)  # Evitar NaN por división por cero

    # Crear figura
    plt.figure(figsize=figsize)

    # Crear heatmap
    sns.heatmap(
        cm,
        annot=True,
        fmt=".2f" if normalize else "d",
        cmap="Blues",
        xticklabels=classes,
        yticklabels=classes,
        cbar=True,
        square=True,
    )

    # Configurar etiquetas
    plt.title(title, fontsize=14, pad=20)
    plt.xlabel("Predicción", fontsize=12)
    plt.ylabel("Real", fontsize=12)

    # Rotar etiquetas
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)

    # Ajustar diseño
    plt.tight_layout()

    # Guardar la figura si se especifica
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    # Mostrar la figura si se solicita
    if show_plot:
        plt.show()
    else:
        plt.close()

    # Imprimir reporte de clasificación
    print("\nReporte de clasificación:")
    print(classification_report(y_true, y_pred, target_names=classes, digits=4))

    return cm


def plot_training_history(
    history: Dict[str, list],
    save_path: Optional[str] = None,
    show_plot: bool = True,
    title: str = "Historial de entrenamiento",
    figsize: Tuple[int, int] = (15, 5),
) -> None:
    """
    Grafica la pérdida y precisión durante el entrenamiento.

    Args:
        history: Diccionario con el historial de entrenamiento. Debe contener al menos
        'train_loss' y 'train_acc'. Opcionalmente puede contener 'val_loss' y 'val_acc'.
        save_path: Ruta para guardar la figura. Si es None, no se guarda.
        show_plot: Si es True, muestra la figura.
        title: Título del gráfico.
        figsize: Tamaño de la figura (ancho, alto).
    """
    # Verificar que el historial contiene las claves necesarias
    required_keys = ["train_loss", "train_acc"]
    for key in required_keys:
        if key not in history:
            raise ValueError(f"El historial debe contener la clave '{key}'")

    epochs = range(1, len(history["train_loss"]) + 1)

    # Crear figura con 2 subgráficos
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    fig.suptitle(title)

    # Gráfico de pérdida
    ax1.plot(history["train_loss"], label="Entrenamiento")
    if "val_loss" in history:
        ax1.plot(history["val_loss"], label="Validación")
    ax1.set_title("Pérdida por época")
    ax1.set_xlabel("Época")
    ax1.set_ylabel("Pérdida")
    ax1.legend()

    # Gráfico de precisión
    ax2.plot(epochs, history["train_acc"], "b-", label="Entrenamiento")
    if "val_acc" in history and history["val_acc"]:
        ax2.plot(epochs, history["val_acc"], "r-", label="Validación")
    ax2.set_title("Precisión por época")
    ax2.set_xlabel("Época")
    ax2.set_ylabel("Precisión")
    ax2.legend()
    ax2.grid(True, linestyle="--", alpha=0.7)

    # Marcar el mejor valor de precisión de validación
    if "val_acc" in history and history["val_acc"]:
        best_val_acc = max(history["val_acc"])
        best_epoch = history["val_acc"].index(best_val_acc) + 1
        ax2.axvline(x=best_epoch, color="gray", linestyle="--", alpha=0.7)
        ax2.text(
            best_epoch,
            min(history["train_acc"] + history["val_acc"]),
            f"Mejor: {best_val_acc:.4f}\nÉpoca: {best_epoch}",
            ha="center",
            va="bottom",
            bbox=dict(facecolor="white", alpha=0.8),
        )

    # Ajustar diseño
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()

    # Guardar la figura si se especifica
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    # Mostrar la figura si se solicita
    if show_plot:
        plt.show()
    else:
        plt.close()


def get_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_names: Optional[List[str]] = None,
    digits: int = 4,
) -> str:
    """
    Genera un reporte de clasificación.

    Args:
        y_true: Etiquetas verdaderas.
        y_pred: Predicciones del modelo.
        target_names: Nombres de las clases.
        digits: Número de decimales para los valores numéricos.

    Returns:
        Reporte de clasificación como cadena de texto.
    """
    return classification_report(
        y_true, y_pred, target_names=target_names, digits=digits, output_dict=False
    )
