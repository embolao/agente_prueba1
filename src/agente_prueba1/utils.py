"""Módulo con utilidades para visualización y manejo de datos."""

import os
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
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


def load_mnist(batch_size: int = 64):
    """
    Carga el dataset MNIST con transformaciones apropiadas.

    Args:
        batch_size: Tamaño del batch para los dataloaders.

    Returns:
        Tuple con (train_loader, val_loader, test_loader, train_dataset, test_dataset)
    """
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms

    # Transformaciones para los datos
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    # Cargar datasets
    train_dataset = datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )

    test_dataset = datasets.MNIST(
        root="./data", train=False, download=True, transform=transform
    )

    # Dividir train en train y validation
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        train_dataset, [train_size, val_size]
    )  # type: ignore  # Ignorar el warning de typing

    # Crear dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, train_dataset, test_dataset


def train_model(
    model,
    train_loader,
    val_loader,
    num_epochs: int,
    learning_rate: float,
    device: str,
    log_dir: str,
    checkpoint_dir: str,
    early_stopping_patience: int = 5,
):
    """
    Entrena el modelo con early stopping y logging.

    Args:
        model: Modelo a entrenar.
        train_loader: DataLoader para entrenamiento.
        val_loader: DataLoader para validación.
        num_epochs: Número de épocas.
        learning_rate: Tasa de aprendizaje.
        device: Dispositivo para el entrenamiento ('cpu' o 'cuda').
        log_dir: Directorio para logs de TensorBoard.
        checkpoint_dir: Directorio para guardar checkpoints.
        early_stopping_patience: Número de épocas sin mejora antes de detenerse.
    """
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.tensorboard import SummaryWriter

    # Configurar TensorBoard
    writer = SummaryWriter(log_dir)

    # Mover modelo al dispositivo
    model = model.to(device)

    # Definir criterio y optimizador
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=model.weight_decay if hasattr(model, "weight_decay") else 0,
    )

    # Historial de entrenamiento
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "best_val_acc": 0,
        "best_epoch": 0,
    }

    # Early stopping
    patience = early_stopping_patience
    no_improvement_count = 0

    for epoch in range(num_epochs):
        # Entrenamiento
        model.train()
        train_loss = 0
        correct = 0
        total = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            data = data.view(data.size(0), -1)  # Aplanar imágenes

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

            if batch_idx % 100 == 0:
                print(
                    f"Epoch [{epoch+1}/{num_epochs}], "
                    f"Step [{batch_idx+1}/{len(train_loader)}], "
                    f"Loss: {loss.item():.4f}"
                )  # Mostrar progreso del entrenamiento

        # Validación
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                data = data.view(data.size(0), -1)

                output = model(data)
                loss = criterion(output, target)

                val_loss += loss.item()
                _, predicted = output.max(1)
                val_total += target.size(0)
                val_correct += predicted.eq(target).sum().item()

        # Calcular métricas
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        train_acc = correct / total
        val_acc = val_correct / val_total

        # Actualizar historial
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        # Logging
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("Accuracy/train", train_acc, epoch)
        writer.add_scalar("Accuracy/val", val_acc, epoch)

        # Early stopping
        if val_acc > history["best_val_acc"]:
            history["best_val_acc"] = val_acc
            history["best_epoch"] = epoch
            torch.save(model.state_dict(), f"{checkpoint_dir}/model_best.pth")
            no_improvement_count = 0
        else:
            no_improvement_count += 1

        if no_improvement_count >= patience:
            print(
                f"Early stopping en epoch {epoch+1} "
                f"(sin mejora en {patience} épocas)"
            )
            break

        print(
            f"Epoch [{epoch+1}/{num_epochs}] - "
            f"Train Loss: {train_loss:.4f}, "
            f"Train Acc: {train_acc:.4f}, "
            f"Val Loss: {val_loss:.4f}, "
            f"Val Acc: {val_acc:.4f}"
        )

    # Cerrar TensorBoard
    writer.close()

    return history


def evaluate_model(model, data_loader, criterion, device: str):
    """
    Evalúa el modelo en un conjunto de datos.

    Args:
        model: Modelo a evaluar.
        data_loader: DataLoader con los datos.
        criterion: Criterio de pérdida.
        device: Dispositivo para la evaluación ('cpu' o 'cuda').

    Returns:
        Tuple con (loss, accuracy)
    """
    model.eval()
    test_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            data = data.view(data.size(0), -1)

            output = model(data)
            loss = criterion(output, target)

            test_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

    test_loss /= len(data_loader)
    accuracy = correct / total

    return test_loss, accuracy
