"""Módulo que contiene la implementación de una red neuronal simple."""

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader


class SimpleNN(nn.Module):
    """
    Una red neuronal mejorada con múltiples capas ocultas.

    Args:
        input_size: Número de características de entrada.
        hidden_sizes: Lista con el número de neuronas en cada capa oculta.
        output_size: Número de neuronas de salida.
        activation: Función de activación a utilizar.
                   Opciones: 'relu', 'leaky_relu', 'sigmoid', 'tanh', 'selu'.
        dropout: Tasa de dropout (None para desactivar).
        batch_norm: Si es True, añade normalización por lotes.
        weight_decay: Coeficiente de decaimiento de pesos (L2).
        use_bias: Si es True, añade sesgo a las capas lineales.
    """

    def __init__(
        self,
        input_size: int,
        hidden_sizes: Optional[List[int]] = None,
        output_size: int = 1,
        activation: str = "relu",
        dropout: float = 0.0,
        batch_norm: bool = False,
        weight_decay: float = 0.0,
        use_bias: bool = True,
    ) -> None:
        if hidden_sizes is None:
            hidden_sizes = [64, 32]
        super().__init__()
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.activation_name = activation.lower()
        self.dropout_rate = dropout
        self.batch_norm = batch_norm
        self.weight_decay = weight_decay
        self.use_bias = use_bias

        # Validar parámetros
        if not isinstance(hidden_sizes, (list, tuple)):
            hidden_sizes = [hidden_sizes]

        # Construir capas
        self.layers = nn.ModuleList()
        prev_size = input_size

        # Añadir capas ocultas
        for i, h_size in enumerate(hidden_sizes):
            self.layers.append(nn.Linear(prev_size, h_size, bias=use_bias))

            # Añadir BatchNorm si está habilitado (excepto en la última capa)
            if batch_norm and i < len(hidden_sizes) - 1:
                self.layers.append(nn.BatchNorm1d(h_size))

            # Añadir activación (excepto en la última capa)
            if i < len(hidden_sizes) - 1:
                self.layers.append(self._get_activation_module(activation))

                # Añadir Dropout si está habilitado
                if dropout is not None:
                    self.layers.append(nn.Dropout(dropout))

            prev_size = h_size

        # Capa de salida
        self.output_layer = nn.Linear(prev_size, output_size, bias=use_bias)

        # Inicialización de pesos
        self._initialize_weights()

        # Función de activación para uso en forward
        self.activation = self._get_activation(activation)

    def _get_activation_module(self, activation: str) -> nn.Module:
        """Obtiene el módulo de activación según el nombre."""
        activations = {
            "relu": nn.ReLU(),
            "leaky_relu": nn.LeakyReLU(0.1),
            "sigmoid": nn.Sigmoid(),
            "tanh": nn.Tanh(),
            "selu": nn.SELU(),
            "none": nn.Identity(),
        }
        return activations.get(activation.lower(), nn.ReLU())

    def _get_activation(self, activation: str) -> callable:
        """Obtiene la función de activación según el nombre (para uso en forward)."""
        activations = {
            "relu": torch.relu,
            "leaky_relu": lambda x: torch.relu(x, 0.1),
            "sigmoid": torch.sigmoid,
            "tanh": torch.tanh,
            "selu": torch.selu,
            "none": lambda x: x,
        }
        return activations.get(activation.lower(), torch.relu)

    def _initialize_weights(self) -> None:
        """Inicializa los pesos de la red."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # Inicialización He (Kaiming) para ReLU/LeakyReLU
                if self.activation_name in ["relu", "leaky_relu"]:
                    nn.init.kaiming_normal_(
                        m.weight, mode="fan_in", nonlinearity="relu"
                    )
                # Inicialización Xavier para tanh/sigmoid
                elif self.activation_name in ["tanh", "sigmoid"]:
                    nn.init.xavier_uniform_(m.weight)
                # Inicialización para SELU
                elif self.activation_name == "selu":
                    nn.init.kaiming_normal_(
                        m.weight, mode="fan_in", nonlinearity="linear"
                    )
                # Inicialización por defecto
                else:
                    nn.init.xavier_uniform_(m.weight)

                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Realiza el paso hacia adelante de la red.

        Args:
            x: Tensor de entrada con forma (batch_size, input_size).

        Returns:
            Tensor de salida con forma (batch_size, output_size).
        """
        for layer in self.layers:
            x = layer(x)
        # Aplicar la capa de salida
        x = self.output_layer(x)
        return x

    def get_config(self) -> Dict[str, Any]:
        """Obtiene la configuración del modelo."""
        return {
            "input_size": self.input_size,
            "hidden_sizes": self.hidden_sizes,
            "output_size": self.output_size,
            "activation": self.activation_name,
            "dropout": self.dropout_rate,
            "batch_norm": self.batch_norm,
            "weight_decay": self.weight_decay,
            "use_bias": self.use_bias,
        }

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "SimpleNN":
        """Crea una instancia del modelo a partir de una configuración."""
        return cls(**config)

    def get_optimizer(
        self, optimizer_class: type = optim.AdamW, **optim_kwargs
    ) -> optim.Optimizer:
        """
        Obtiene un optimizador configurado para este modelo.

        Args:
            optimizer_class: Clase del optimizador a utilizar (por defecto: AdamW).
            **optim_kwargs: Argumentos adicionales para el optimizador.

        Returns:
            Optimizador configurado con los parámetros del modelo.
        """
        return optimizer_class(
            self.parameters(), weight_decay=self.weight_decay, **optim_kwargs
        )

    def get_lr_scheduler(
        self,
        optimizer: optim.Optimizer,
        scheduler_class: type = optim.lr_scheduler.StepLR,
        **scheduler_kwargs,
    ) -> _LRScheduler:
        """
        Obtiene un programador de tasa de aprendizaje.

        Args:
            optimizer: Optimizador para el cual configurar el scheduler.
            scheduler_class: Clase del scheduler a utilizar (por defecto: StepLR).
            **scheduler_kwargs: Argumentos adicionales para el scheduler.

        Returns:
            Scheduler configurado para el optimizador.
        """
        # Configuración por defecto para StepLR
        if (
            scheduler_class == optim.lr_scheduler.StepLR
            and "step_size" not in scheduler_kwargs
        ):
            scheduler_kwargs["step_size"] = 5
        if "gamma" not in scheduler_kwargs:
            scheduler_kwargs["gamma"] = 0.1

        return scheduler_class(optimizer, **scheduler_kwargs)


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader] = None,
    criterion: Optional[nn.Module] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    num_epochs: int = 10,
    device: Optional[str] = None,
    verbose: int = 1,
) -> Dict[str, List[float]]:
    """Entrena un modelo de PyTorch.

    Args:
        model: Modelo a entrenar
        train_loader: DataLoader para datos de entrenamiento
        val_loader: DataLoader opcional para validación
        criterion: Función de pérdida
        optimizer: Optimizador a utilizar
        scheduler: Programador de tasa de aprendizaje
        num_epochs: Número de épocas de entrenamiento
        device: Dispositivo a utilizar ('cuda' o 'cpu')
        verbose: Nivel de verbosidad (0, 1 o 2)

    Returns:
        Diccionario con el historial de entrenamiento
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    # Configuración por defecto
    if criterion is None:
        criterion = nn.CrossEntropyLoss()

    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Historial de entrenamiento
    history = {
        "train_loss": [],
        "train_acc": [],
    }

    if val_loader is not None:
        history["val_loss"] = []
        history["val_acc"] = []

    for epoch in range(num_epochs):
        # Modo entrenamiento
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        # Barra de progreso
        if verbose == 1:
            from tqdm import tqdm

            train_iter = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        else:
            train_iter = train_loader

        for inputs, labels in train_iter:
            inputs, labels = inputs.to(device), labels.to(device)

            # Cero los gradientes
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass y optimización
            loss.backward()
            optimizer.step()

            # Actualizar el scheduler si está definido
            if scheduler is not None and isinstance(
                scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
            ):
                # Para ReduceLROnPlateau, necesitamos la métrica de validación
                if (
                    val_loader is not None
                    and "val_loss" in history
                    and history["val_loss"]
                ):
                    scheduler.step(history["val_loss"][-1])

            # Estadísticas
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            if verbose == 1:
                current_lr = optimizer.param_groups[0]["lr"]
                train_iter.set_postfix(
                    {
                        "loss": running_loss / total,
                        "acc": 100 * correct / total,
                        "lr": f"{current_lr:.2e}",
                    }
                )

        # Guardar métricas de entrenamiento
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = 100 * correct / total
        history["train_loss"].append(epoch_loss)
        history["train_acc"].append(epoch_acc)

        # Validación
        if val_loader is not None:
            val_loss, val_acc = evaluate_model(model, val_loader, criterion, device)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)

            # Actualizar el scheduler después de la validación (para StepLR, etc.)
            if scheduler is not None and not isinstance(
                scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
            ):
                scheduler.step()

            if verbose > 0:
                current_lr = optimizer.param_groups[0]["lr"]
                print(
                    f"Epoch {epoch+1}/{num_epochs} - "
                    f"Loss: {epoch_loss:.4f} - "
                    f"Acc: {epoch_acc:.2f}% - "
                    f"Val Loss: {val_loss:.4f} - "
                    f"Val Acc: {val_acc:.2f}% - "
                    f"LR: {current_lr:.2e}"
                )
        elif verbose > 0:
            current_lr = optimizer.param_groups[0]["lr"]
            print(
                f"Epoch {epoch+1}/{num_epochs} - "
                f"Loss: {epoch_loss:.4f} - "
                f"Acc: {epoch_acc:.2f}% - "
                f"LR: {current_lr:.2e}"
            )

    return history


def evaluate_model(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: Optional[nn.Module] = None,
    device: Optional[str] = None,
) -> Tuple[float, float]:
    """Evalúa un modelo en un DataLoader dado.

    Args:
        model: Modelo a evaluar
        data_loader: DataLoader con los datos de evaluación
        criterion: Función de pérdida
        device: Dispositivo a utilizar ('cuda' o 'cpu')

        criterion: Función de pérdida.
        device: Dispositivo a utilizar ('cuda' o 'cpu').

    Returns:
        Tupla con (pérdida, precisión).
    """
    if criterion is None:
        criterion = nn.CrossEntropyLoss()

    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    loss = running_loss / len(data_loader.dataset)
    acc = correct / total  # Devolver en rango [0, 1]

    return loss, acc
