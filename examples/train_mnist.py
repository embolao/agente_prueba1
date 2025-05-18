"""
Ejemplo de entrenamiento de una red neuronal en el conjunto de datos MNIST.
"""

import os
import sys
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from tqdm import tqdm

# Añadir el directorio raíz al path para importaciones
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# Configurar estilo de las gráficas
plt.style.use("ggplot")
sns.set_palette("husl")


def load_mnist(batch_size=64, val_split=0.2):
    """
    Carga el conjunto de datos MNIST.

    Args:
        batch_size: Tamaño del lote.
        val_split: Proporción de datos para validación.

    Returns:
        Tupla con (train_loader, val_loader, test_loader, input_size, num_classes).
    """
    # Transformaciones para los datos
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
            transforms.Lambda(lambda x: x.view(-1)),  # Aplanar la imagen
        ]
    )

    # Descargar y cargar el conjunto de entrenamiento
    train_val_dataset = datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )

    # Dividir en entrenamiento y validación
    val_size = int(val_split * len(train_val_dataset))
    train_size = len(train_val_dataset) - val_size
    train_dataset, val_dataset = random_split(train_val_dataset, [train_size, val_size])

    # Cargar conjunto de prueba
    test_dataset = datasets.MNIST(
        root="./data", train=False, download=True, transform=transform
    )

    # Crear DataLoaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=2
    )

    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=2
    )

    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=2
    )

    # Tamaño de entrada (28x28 = 784 píxeles)
    input_size = 28 * 28
    num_classes = 10  # MNIST tiene 10 clases (dígitos del 0 al 9)

    return train_loader, val_loader, test_loader, input_size, num_classes


def plot_confusion_matrix(model, data_loader, device, class_names=None):
    """
    Calcula y grafica la matriz de confusión para un modelo y un DataLoader dados.

    Args:
        model: Modelo PyTorch ya entrenado.
        data_loader: DataLoader con los datos de validación o prueba.
        device: Dispositivo donde se ejecutará el modelo.
        class_names: Lista con los nombres de las clases.
    """
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Calcular matriz de confusión
    cm = confusion_matrix(all_labels, all_preds)

    # Normalizar por filas (porcentaje por clase real)
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    cm_norm = np.nan_to_num(cm_norm)  # Evitar NaN por división por cero

    # Configurar etiquetas de clases
    if class_names is None:
        class_names = [str(i) for i in range(cm.shape[0])]

    # Crear figura
    plt.figure(figsize=(12, 10))

    # Crear heatmap
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=True,
    )

    # Configurar etiquetas
    plt.title("Matriz de confusión normalizada", fontsize=14, pad=20)
    plt.xlabel("Predicción", fontsize=12)
    plt.ylabel("Real", fontsize=12)

    # Rotar etiquetas
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)

    # Ajustar diseño
    plt.tight_layout()

    # Guardar la figura
    os.makedirs("figures", exist_ok=True)
    plt.savefig("figures/confusion_matrix.png", dpi=300, bbox_inches="tight")
    plt.show()

    # Imprimir reporte de clasificación
    print("\nReporte de clasificación:")
    print(
        classification_report(
            all_labels,
            all_preds,
            target_names=class_names if class_names else None,
            digits=4,
        )
    )

    return cm


def plot_training_history(history, val_loader, device):
    """
    Grafica la pérdida, precisión y tasa de aprendizaje durante el entrenamiento.

    Args:
        history: Diccionario con el historial de entrenamiento.
        val_loader: DataLoader para el conjunto de validación.
        device: Dispositivo donde se ejecuta el modelo.
    """
    sns.set_style("whitegrid")

    # Crear figura con 3 subgráficos
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

    # Gráfico de pérdida
    epochs = range(1, len(history["train_loss"]) + 1)
    ax1.plot(epochs, history["train_loss"], "b-", label="Entrenamiento")
    if "val_loss" in history and history["val_loss"]:
        ax1.plot(epochs, history["val_loss"], "r-", label="Validación")
    ax1.set_title("Pérdida por época")
    ax1.set_xlabel("Época")
    ax1.set_ylabel("Pérdida")
    ax1.legend()
    ax1.grid(True, linestyle="--", alpha=0.7)

    # Marcar el mejor valor de pérdida de validación si está disponible
    if "val_loss" in history and history["val_loss"]:
        best_epoch = np.argmin(history["val_loss"])
        best_loss = history["val_loss"][best_epoch]
        ax1.plot(best_epoch + 1, best_loss, "ro")
        ax1.annotate(
            f"Mínimo: {best_loss:.4f}",
            xy=(best_epoch + 1, best_loss),
            xytext=(best_epoch + 1, best_loss * 1.1),
            arrowprops=dict(facecolor="black", shrink=0.05),
            horizontalalignment="center",
        )

    # Gráfico de precisión
    ax2.plot(epochs, history["train_acc"], "b-", label="Entrenamiento")
    if "val_acc" in history and history["val_acc"]:
        ax2.plot(epochs, history["val_acc"], "r-", label="Validación")
    ax2.set_title("Precisión por época")
    ax2.set_xlabel("Época")
    ax2.set_ylabel("Precisión (%)")
    ax2.legend()
    ax2.grid(True, linestyle="--", alpha=0.7)

    # Marcar la mejor precisión de validación si está disponible
    if "val_acc" in history and history["val_acc"]:
        best_acc = max(history["val_acc"])
        best_acc_epoch = np.argmax(history["val_acc"])
        ax2.plot(best_acc_epoch + 1, best_acc, "ro")
        ax2.annotate(
            f"Máximo: {best_acc:.2f}%",
            xy=(best_acc_epoch + 1, best_acc),
            xytext=(best_acc_epoch + 1, best_acc * 0.9),
            arrowprops=dict(facecolor="black", shrink=0.05),
            horizontalalignment="center",
        )

    # Gráfico de tasa de aprendizaje
    if "learning_rates" in history and len(history["learning_rates"]) > 0:
        ax3.plot(epochs, history["learning_rates"], "g-")
        ax3.set_title("Tasa de aprendizaje")
        ax3.set_xlabel("Época")
        ax3.set_ylabel("Learning Rate")
        ax3.set_yscale("log")
        ax3.grid(True, linestyle="--", alpha=0.7)

    plt.tight_layout()

    # Guardar la figura
    os.makedirs("figures", exist_ok=True)
    plt.savefig("figures/training_history.png", dpi=300, bbox_inches="tight")
    plt.show()

    # Mostrar matriz de confusión en el conjunto de validación si hay un modelo
    if "model" in history and val_loader is not None:
        plot_confusion_matrix(history["model"], val_loader, device)


def train_mnist_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int = 10,
    learning_rate: float = 0.001,
    device: Optional[str] = None,
    log_dir: Optional[str] = None,
    checkpoint_dir: Optional[str] = None,
    early_stopping_patience: Optional[int] = None,
) -> Dict[str, List[float]]:
    """
    Entrena el modelo con características avanzadas como early stopping,
    checkpoints y logging.

    Args:
        model: Modelo a entrenar.
        train_loader: DataLoader para datos de entrenamiento.
        val_loader: DataLoader para datos de validación.
        num_epochs: Número de épocas de entrenamiento.
        learning_rate: Tasa de aprendizaje inicial.
        device: Dispositivo para entrenamiento ('cuda' o 'cpu').
        log_dir: Directorio para guardar logs de TensorBoard.
        checkpoint_dir: Directorio para guardar checkpoints del modelo.
        early_stopping_patience: Número de épocas para esperar antes de
        detener el entrenamiento
        si no mejora la pérdida de validación. None para desactivar.

    Returns:
        Diccionario con el historial de entrenamiento y el mejor modelo.
    """
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()

    # Inicializar el optimizador
    if hasattr(model, "get_optimizer"):
        optimizer = model.get_optimizer(lr=learning_rate)
    else:
        # Usar Adam con weight decay
        weight_decay = getattr(model, "weight_decay", 1e-4)
        optimizer = torch.optim.Adam(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )

    # Programador de tasa de aprendizaje
    if hasattr(model, "get_lr_scheduler"):
        scheduler = model.get_lr_scheduler(optimizer)
    else:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, "min", patience=3, factor=0.5, verbose=True
        )

    # Configurar directorios
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        writer = None
        if log_dir:
            from torch.utils.tensorboard import SummaryWriter

            writer = SummaryWriter(log_dir)
    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)

    # Inicializar el historial
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "lr": [],
    }

    # Early stopping
    best_val_loss = float("inf")
    best_val_acc = 0.0
    epochs_no_improve = 0
    best_model_wts = None

    # Bucle de entrenamiento
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        # Barra de progreso
        train_loop = tqdm(
            train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False
        )

        for inputs, labels in train_loop:
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Añadir regularización L2 si está habilitada
            if hasattr(model, "l2_regularization"):
                l2_lambda = model.l2_regularization
                l2_norm = sum(p.pow(2.0).sum() for p in model.parameters())
                loss = loss + l2_lambda * l2_norm

            # Backward y optimización
            optimizer.zero_grad()
            loss.backward()

            # Aplicar recorte de gradiente si está disponible
            if hasattr(model, "clip_gradients"):
                model.clip_gradients()

            optimizer.step()

            # Estadísticas
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Actualizar barra de progreso
            train_loop.set_postfix(
                {
                    "loss": running_loss / total,
                    "acc": 100 * correct / total,
                }
            )

        # Calcular métricas de entrenamiento
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = 100 * correct / total

        # Validación
        val_loss, val_acc = evaluate_mnist_model(
            model=model, data_loader=val_loader, criterion=criterion, device=device
        )

        # Actualizar el scheduler tras validación
        if scheduler is not None and not isinstance(
            scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
        ):
            scheduler.step(val_loss)
        else:
            scheduler.step()

        # Guardar métricas
        current_lr = optimizer.param_groups[0]["lr"]
        history["train_loss"].append(epoch_loss)
        history["train_acc"].append(epoch_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["lr"].append(current_lr)

        # Log en TensorBoard
        if log_dir:
            writer.add_scalar("Loss/train", epoch_loss, epoch)
            writer.add_scalar("Loss/val", val_loss, epoch)
            writer.add_scalar("Accuracy/train", epoch_acc, epoch)
            writer.add_scalar("Accuracy/val", val_acc, epoch)
            writer.add_scalar("Learning Rate", current_lr, epoch)

        # Guardar checkpoint
        if checkpoint_dir:
            checkpoint_path = os.path.join(
                checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pth"
            )
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                },
                checkpoint_path,
            )

        # Early stopping basado en la pérdida de validación
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            best_model_wts = model.state_dict().copy()
            epochs_no_improve = 0

            # Guardar el mejor modelo
            if checkpoint_dir:
                best_model_path = os.path.join(checkpoint_dir, "best_model.pth")
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": best_model_wts,
                        "optimizer_state_dict": optimizer.state_dict(),
                        "val_loss": best_val_loss,
                        "val_acc": best_val_acc,
                    },
                    best_model_path,
                )
        else:
            epochs_no_improve += 1
            if (
                early_stopping_patience is not None
                and epochs_no_improve >= early_stopping_patience
            ):
                print(f"\nEarly stopping después de {epoch+1} épocas sin mejora")
                break

        # Imprimir métricas
        print(
            f"Epoch [{epoch+1}/{num_epochs}], "
            f"Train Loss: {epoch_loss:.4f}, "
            f"Train Acc: {epoch_acc:.2f}%, "
            f"Val Loss: {val_loss:.4f}, "
            f"Val Acc: {val_acc:.2f}%, "
            f"LR: {scheduler.get_last_lr()[0]:.6f}"
            if scheduler
            else ""
        )

    # Cerrar el escritor de TensorBoard
    if log_dir:
        writer.close()

    # Cargar los mejores pesos
    if best_model_wts is not None:
        model.load_state_dict(best_model_wts)

    return {
        "model": model,
        "history": history,
        "best_val_loss": best_val_loss,
        "best_val_acc": best_val_acc,
    }


def evaluate_mnist_model(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: Optional[nn.Module] = None,
    device: Optional[str] = None,
) -> Tuple[float, float]:
    """
    Evalúa el modelo en un DataLoader dado.

    Args:
        model: Modelo a evaluar.
        data_loader: DataLoader con los datos de evaluación.
        criterion: Función de pérdida (opcional).
        device: Dispositivo para la evaluación.

    Returns:
        Si se proporciona criterion: tupla con (pérdida promedio, precisión).
        Si no se proporciona criterion: precisión.
    """
    model.eval()

    # Si no se proporciona una función de pérdida, solo calculamos la precisión
    calculate_loss = criterion is not None

    running_loss = 0.0 if calculate_loss else None
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)

            if calculate_loss and criterion is not None:
                loss = criterion(outputs, labels)
                running_loss += loss.item() * inputs.size(0)

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total

    if calculate_loss and running_loss is not None:
        avg_loss = running_loss / len(data_loader.dataset)
        return avg_loss, accuracy

    return accuracy


def main():
    # Configuración
    batch_size = 64
    num_epochs = 10
    learning_rate = 1e-3
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_path = "models/mnist_model.pth"

    # Crear directorios si no existen
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    # Cargar datos
    train_loader, val_loader, test_loader, input_size, num_classes = load_mnist(
        batch_size=batch_size
    )

    # Configuración del modelo
    config = {
        "input_size": input_size,
        "hidden_sizes": [256, 128, 64],
        "output_size": num_classes,
        "activation": "relu",
        "dropout": 0.2,
        "batch_norm": True,
        "weight_decay": 1e-4,
        "use_bias": True,
    }

    # Crear modelo
    from agente_prueba1.model import SimpleNN

    model = SimpleNN(**config)
    model = model.to(device)
    print("\nModelo creado:")
    print(model)

    # Entrenar el modelo
    print("\nEntrenando el modelo...")
    training_result = train_mnist_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        device=device,
        log_dir="logs/mnist_experiment",
        checkpoint_dir="checkpoints",
        early_stopping_patience=5,
    )

    # Obtener el mejor modelo y el historial
    best_model = training_result["model"]
    history = training_result["history"]

    # Graficar historial de entrenamiento
    plot_training_history(history, val_loader, device)

    # Evaluar en el conjunto de prueba
    print("\nEvaluando en el conjunto de prueba...")
    test_loss, test_acc = evaluate_mnist_model(
        model=best_model,
        data_loader=test_loader,
        criterion=nn.CrossEntropyLoss(),
        device=device,
    )
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.2f}%")

    # Guardar el mejor modelo
    torch.save(
        {
            "model_state_dict": best_model.state_dict(),
            "config": config,
            "test_accuracy": test_acc,
            "training_history": history,
        },
        model_path,
    )

    # Guardar la configuración en un archivo separado
    with open("models/model_config.json", "w") as f:
        import json

        json.dump(config, f, indent=2)

    print(f"\nModelo guardado en {model_path}")
    print("Configuración guardada en models/model_config.json")


if __name__ == "__main__":
    main()
