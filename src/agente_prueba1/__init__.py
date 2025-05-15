"""Agente de prueba 1: Implementación de una red neuronal simple."""

from .model import SimpleNN, evaluate_model, train_model
from .utils import (
    get_classification_report,
    plot_confusion_matrix,
    plot_training_history,
)

__all__ = [
    "SimpleNN",
    "train_model",
    "evaluate_model",
    "plot_confusion_matrix",
    "plot_training_history",
    "get_classification_report",
]

# Versión del paquete
try:
    from importlib.metadata import PackageNotFoundError, version

    __version__ = version("agente_prueba1")
except (ImportError, PackageNotFoundError):
    __version__ = "0.1.0"
