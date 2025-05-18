"""Pruebas de integración para el flujo completo de entrenamiento y evaluación."""

import os
import tempfile

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


def test_full_training_workflow():
    """Prueba el flujo completo de carga de datos, entrenamiento y evaluación."""
    from agente_prueba1 import SimpleNN, evaluate_model, train_model

    # Crear datos de ejemplo y moverlos a CUDA
    device = torch.device("cuda")
    x_train = torch.randn(100, 784, device=device)  # 100 muestras, 784 características
    y_train = torch.randint(0, 10, (100,), device=device)  # 100 etiquetas (10 clases)
    x_val = torch.randn(20, 784, device=device)  # 20 muestras de validación
    y_val = torch.randint(0, 10, (20,), device=device)  # 20 etiquetas de validación

    # Crear datasets y dataloaders
    train_dataset = TensorDataset(x_train, y_train)
    val_dataset = TensorDataset(x_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=10, shuffle=False)

    # Crear modelo y forzar uso de CPU
    model = SimpleNN(
        input_size=784,
        hidden_sizes=[64, 32],
        output_size=10,
        activation="relu",
        dropout=0.2,
        batch_norm=True,
        weight_decay=1e-4,
        use_bias=True,
    ).to("cpu")

    # Función de pérdida
    criterion = nn.CrossEntropyLoss()

    # Configurar el optimizador
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)

    # Configurar el scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    # Entrenar el modelo
    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=2,  # Solo 2 épocas para pruebas rápidas
        verbose=0,
    )

    # Verificar que el historial contiene las claves esperadas
    expected_keys = ["train_loss", "val_loss", "train_acc", "val_acc"]
    assert all(key in history for key in expected_keys)

    # Verificar que las listas de historial tienen la longitud correcta
    assert len(history["train_loss"]) == 2
    assert len(history["val_loss"]) == 2
    assert len(history["train_acc"]) == 2
    assert len(history["val_acc"]) == 2

    # Evaluar el modelo
    val_loss, val_accuracy = evaluate_model(
        model=model, data_loader=val_loader, criterion=criterion
    )

    # Verificar que se devolvieron valores numéricos
    assert isinstance(val_loss, float)
    assert isinstance(val_accuracy, float)

    # Verificar que la precisión está en el rango correcto
    assert 0.0 <= val_accuracy <= 1.0

    # Verificar que la pérdida es un número finito
    assert not torch.isnan(torch.tensor(val_loss)).any()
    assert not torch.isinf(torch.tensor(val_loss)).any()

    # Probar guardar y cargar el modelo
    with tempfile.TemporaryDirectory() as temp_dir:
        model_path = os.path.join(temp_dir, "test_model.pth")

        # Guardar el modelo
        torch.save(
            {"model_state_dict": model.state_dict(), "config": model.get_config()},
            model_path,
        )

        # Cargar el modelo
        device = next(
            model.parameters()
        ).device  # Obtener el dispositivo del modelo original
        checkpoint = torch.load(model_path, map_location=device)
        loaded_model = SimpleNN.from_config(checkpoint["config"]).to(device)
        loaded_model.load_state_dict(checkpoint["model_state_dict"])

        # Verificar que el modelo cargado produce la misma salida
        model.eval()
        loaded_model.eval()
        with torch.no_grad():
            # Mover los datos al dispositivo correcto
            x_val_device = x_val.to(device)
            output1 = model(x_val_device)
            output2 = loaded_model(x_val_device)
            assert torch.allclose(output1, output2, atol=1e-6)
