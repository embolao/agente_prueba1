"""Pruebas para las funciones de utilidad."""

import os
import tempfile

import matplotlib.pyplot as plt
import numpy as np
import pytest
from torch.utils.tensorboard import SummaryWriter


def test_plot_training_history():
    """Prueba la función plot_training_history."""
    from agente_prueba1.utils import plot_training_history

    # Crear datos de ejemplo
    history = {
        "train_loss": [0.8, 0.6, 0.5, 0.4],
        "val_loss": [0.9, 0.7, 0.6, 0.5],
        "train_acc": [0.7, 0.75, 0.8, 0.85],
        "val_acc": [0.65, 0.72, 0.78, 0.82],
    }

    # Usar un directorio temporal con el manejador de contexto
    with tempfile.TemporaryDirectory() as temp_dir:
        output_path = os.path.join(temp_dir, "test_plot.png")

        # Llamar a la función
        plot_training_history(
            history=history,
            save_path=output_path,
            show_plot=False,
            title="Test Plot",
            figsize=(12, 5),
        )

        # Verificar que se creó el archivo
        assert os.path.exists(output_path)

        # Verificar que la figura se puede cargar
        try:
            img = plt.imread(output_path)
            assert img is not None
        except Exception as e:
            pytest.fail(f"Error al cargar la imagen: {e}")

    # Probar sin validación
    history_no_val = {
        "train_loss": [0.8, 0.6, 0.5, 0.4],
        "train_acc": [0.7, 0.75, 0.8, 0.85],
    }

    # Llamada sin guardar archivo
    plot_training_history(history=history_no_val, save_path=None, show_plot=False)

    # Probar con error
    with pytest.raises(ValueError):
        plot_training_history(history={"invalid_key": [1, 2, 3]})


def test_plot_confusion_matrix():
    """Prueba la función plot_confusion_matrix."""
    from agente_prueba1.utils import plot_confusion_matrix

    # Crear datos de ejemplo
    y_true = np.random.randint(0, 10, size=100)
    y_pred = np.random.randint(0, 10, size=100)

    # Usar un directorio temporal con el manejador de contexto
    with tempfile.TemporaryDirectory() as temp_dir:
        output_path = os.path.join(temp_dir, "confusion_matrix.png")

        # Llamar a la función
        cm = plot_confusion_matrix(
            y_true=y_true,
            y_pred=y_pred,
            classes=[str(i) for i in range(10)],
            save_path=output_path,
            show_plot=False,
            normalize=True,
        )

        # Verificar que se devolvió la matriz de confusión
        assert isinstance(cm, np.ndarray)
        assert cm.shape == (10, 10)  # 10 clases

        # Verificar que se creó el archivo
        assert os.path.exists(output_path)

        # Verificar que la figura se puede cargar
        try:
            img = plt.imread(output_path)
            assert img is not None
        except Exception as e:
            pytest.fail(f"Error al cargar la imagen: {e}")


def test_tensorboard_logging():
    """Prueba el logging con TensorBoard."""
    # Crear un directorio temporal para los logs
    with tempfile.TemporaryDirectory() as log_dir:
        # Crear un escritor de Summary
        writer = SummaryWriter(log_dir=log_dir)

        try:
            # Escribir algunos valores
            for i in range(5):
                writer.add_scalar("test/loss", 1.0 / (i + 1), i)
                writer.add_scalar("test/accuracy", 0.1 * i, i)
        finally:
            # Asegurarse de que el escritor se cierre
            writer.close()

        # Verificar que se crearon los archivos de evento
        event_files = [f for f in os.listdir(log_dir) if "events.out.tfevents" in f]
        assert len(event_files) > 0
