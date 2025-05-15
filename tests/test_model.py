"""Pruebas para el módulo model.py."""

import numpy as np
import torch
import torch.nn as nn


def test_simplenn_initialization():
    """Prueba la inicialización de SimpleNN."""
    from agente_prueba1 import SimpleNN

    # Configuración de prueba
    input_size = 784
    hidden_sizes = [256, 128, 64]
    output_size = 10

    # Crear modelo
    model = SimpleNN(
        input_size=input_size,
        hidden_sizes=hidden_sizes,
        output_size=output_size,
        activation="relu",
        dropout=0.2,
        batch_norm=True,
        weight_decay=1e-4,
        use_bias=True,
    )

    # Verificar atributos
    assert model.input_size == input_size
    assert model.hidden_sizes == hidden_sizes
    assert model.output_size == output_size
    assert model.activation_name == "relu"
    assert model.dropout_rate == 0.2
    assert model.batch_norm is True
    assert model.weight_decay == 1e-4
    assert model.use_bias is True

    # Verificar que todas las capas se crearon correctamente
    # Para hidden_sizes=[256, 128, 64], esperamos 9 capas en total:
    # Para cada capa oculta: Linear -> BatchNorm -> ReLU -> Dropout
    # Para la última capa: Linear
    # Total: 3 capas ocultas * (Linear + BatchNorm + ReLU + Dropout) - 1
    # 3*(Linear + BatchNorm + ReLU + Dropout) - 1 (sin Dropout en la última capa)
    assert len(list(model.layers)) == 9
    assert isinstance(model.output_layer, nn.Linear)


def test_simplenn_forward(simple_model, sample_data):
    """Prueba el paso hacia adelante de SimpleNN."""
    x, _ = sample_data

    # Pasar datos a través del modelo
    output = simple_model(x)

    # Verificar dimensiones de salida
    assert output.shape == (x.shape[0], 10)  # 100 muestras, 10 clases

    # Verificar que no hay NaN o infinitos
    assert not torch.isnan(output).any()
    assert not torch.isinf(output).any()


def test_simplenn_get_config(simple_model):
    """Prueba el método get_config de SimpleNN."""
    config = simple_model.get_config()

    # Verificar que la configuración contiene todas las claves esperadas
    expected_keys = [
        "input_size",
        "hidden_sizes",
        "output_size",
        "activation",
        "dropout",
        "batch_norm",
        "weight_decay",
        "use_bias",
    ]
    assert all(key in config for key in expected_keys)

    # Verificar que los valores coinciden
    assert config["input_size"] == 784
    assert config["hidden_sizes"] == [128, 64]
    assert config["output_size"] == 10
    assert config["activation"] == "relu"
    assert config["dropout"] == 0.2
    assert config["batch_norm"] is True
    assert config["weight_decay"] == 1e-4
    assert config["use_bias"] is True


def test_simplenn_from_config():
    """Prueba el método from_config de SimpleNN."""
    from agente_prueba1 import SimpleNN

    # Configuración de prueba
    config = {
        "input_size": 100,
        "hidden_sizes": [50, 25],
        "output_size": 5,
        "activation": "leaky_relu",
        "dropout": 0.3,
        "batch_norm": False,
        "weight_decay": 1e-5,
        "use_bias": False,
    }

    # Crear modelo a partir de la configuración
    model = SimpleNN.from_config(config)

    # Verificar que la configuración se aplicó correctamente
    assert model.input_size == config["input_size"]
    assert model.hidden_sizes == config["hidden_sizes"]
    assert model.output_size == config["output_size"]
    assert model.activation_name == config["activation"]
    assert model.dropout_rate == config["dropout"]
    assert model.batch_norm == config["batch_norm"]
    assert model.weight_decay == config["weight_decay"]
    assert model.use_bias == config["use_bias"]


def test_simplenn_get_optimizer(simple_model):
    """Prueba el método get_optimizer de SimpleNN."""
    lr = 0.01
    optimizer = simple_model.get_optimizer(lr=lr)

    # Verificar que se creó un optimizador AdamW
    assert isinstance(optimizer, torch.optim.AdamW)

    # Verificar que la tasa de aprendizaje y weight_decay son correctas
    assert optimizer.param_groups[0]["lr"] == lr
    assert optimizer.param_groups[0]["weight_decay"] == simple_model.weight_decay


def test_simplenn_get_lr_scheduler(simple_model):
    """Prueba el método get_lr_scheduler de SimpleNN."""
    optimizer = torch.optim.SGD(simple_model.parameters(), lr=0.01)
    step_size = 5
    gamma = 0.1

    scheduler = simple_model.get_lr_scheduler(
        optimizer, step_size=step_size, gamma=gamma
    )

    # Verificar que se creó un scheduler StepLR
    assert isinstance(scheduler, torch.optim.lr_scheduler.StepLR)
    assert scheduler.step_size == step_size
    assert scheduler.gamma == gamma


def test_train_model(simple_model, data_loader):
    """Prueba la función train_model."""
    from agente_prueba1 import train_model

    # Configurar el modelo y el optimizador
    model = simple_model
    criterion = nn.CrossEntropyLoss()

    # Entrenar por una época
    history = train_model(
        model=model,
        train_loader=data_loader,
        val_loader=None,
        criterion=criterion,
        num_epochs=1,
        verbose=0,
    )

    # Verificar que se devolvió un diccionario con las claves esperadas
    expected_keys = ["train_loss", "train_acc"]
    assert all(key in history for key in expected_keys)

    # Verificar que las listas de historial tienen la longitud correcta
    assert len(history["train_loss"]) == 1
    assert len(history["train_acc"]) == 1


def test_evaluate_model(simple_model, data_loader, device):
    """Prueba la función evaluate_model."""
    from agente_prueba1 import evaluate_model

    # Mover el modelo al dispositivo correcto
    model = simple_model.to(device)

    # Evaluar el modelo
    criterion = nn.CrossEntropyLoss()
    loss, accuracy = evaluate_model(
        model=model, data_loader=data_loader, criterion=criterion, device=device
    )

    # Verificar que se devolvieron valores numéricos
    assert isinstance(loss, float)
    assert isinstance(accuracy, float)

    # Verificar que la precisión está en el rango correcto
    assert 0.0 <= accuracy <= 1.0

    # Verificar que la pérdida es un número finito
    assert not np.isnan(loss)
    assert not np.isinf(loss)
