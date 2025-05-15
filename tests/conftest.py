"""Configuración de pytest para agente_prueba1.

Este archivo contiene configuraciones y fixtures comunes para las pruebas.
"""

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

# Configuración de semillas para reproducibilidad
torch.manual_seed(42)


@pytest.fixture(scope="session")
def device():
    """Dispositivo a utilizar para las pruebas (GPU si está disponible, si no CPU)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def sample_data():
    """Genera datos de ejemplo para pruebas."""
    # Generar datos de ejemplo (MNIST-like)
    x = torch.randn(100, 28 * 28)  # 100 muestras, 784 características
    y = torch.randint(0, 10, (100,))  # 10 clases
    return x, y


@pytest.fixture
def data_loader(sample_data):
    """Crea un DataLoader con datos de ejemplo."""
    x, y = sample_data
    dataset = TensorDataset(x, y)
    return DataLoader(dataset, batch_size=10, shuffle=True)


@pytest.fixture
def simple_model():
    """Crea una instancia de SimpleNN para pruebas."""
    from agente_prueba1 import SimpleNN

    return SimpleNN(
        input_size=784,
        hidden_sizes=[128, 64],
        output_size=10,
        activation="relu",
        dropout=0.2,
        batch_norm=True,
        weight_decay=1e-4,
        use_bias=True,
    )
