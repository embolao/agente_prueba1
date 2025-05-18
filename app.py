"""
Aplicación para entrenar y evaluar un modelo de clasificación de MNIST.
"""

import argparse
import json
import os
from datetime import datetime

import torch
from torch.utils.tensorboard import SummaryWriter

from agente_prueba1.model import SimpleNN

def create_parser():
    """Crea el parser de argumentos."""
    parser = argparse.ArgumentParser(description="Entrenamiento y evaluación de MNIST")
    
    # Modo de operación
    parser.add_argument(
        "mode",
        choices=["train", "evaluate", "predict"],
        help="Modo de operación: train (entrenar), evaluate (evaluar), predict (predecir)"
    )
    
    # Parámetros comunes
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Dispositivo para el entrenamiento (cpu o cuda)"
    )
    
    # Parámetros para entrenamiento
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Número de épocas de entrenamiento"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Tamaño del batch"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.001,
        help="Tasa de aprendizaje"
    )
    
    # Parámetros para la arquitectura del modelo
    parser.add_argument(
        "--hidden-sizes",
        type=int,
        nargs="+",
        default=[256, 128, 64],
        help="Tamaños de las capas ocultas"
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.2,
        help="Probabilidad de dropout"
    )
    parser.add_argument(
        "--activation",
        choices=["relu", "sigmoid", "tanh"],
        default="relu",
        help="Función de activación"
    )
    
    # Parámetros para evaluación y predicción
    parser.add_argument(
        "--model-path",
        default="models/mnist_model.pth",
        help="Ruta del modelo a cargar"
    )
    parser.add_argument(
        "--config-path",
        default="models/model_config.json",
        help="Ruta de la configuración del modelo"
    )
    
    return parser

def train_model(args):
    """Modo de entrenamiento."""
    from examples.train_mnist import train_mnist_model, load_mnist
    
    # Configuración del modelo
    config = {
        "input_size": 784,
        "hidden_sizes": args.hidden_sizes,
        "output_size": 10,
        "activation": args.activation,
        "dropout": args.dropout,
        "batch_norm": True,
        "weight_decay": 1e-4,
        "use_bias": True
    }
    
    # Cargar datos
    train_loader, val_loader, test_loader, _, _ = load_mnist(
        batch_size=args.batch_size
    )
    
    # Crear modelo
    model = SimpleNN(**config)
    model = model.to(args.device)
    
    # Entrenar modelo
    history = train_mnist_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.epochs,
        learning_rate=args.learning_rate,
        device=args.device,
        log_dir=f"runs/mnist_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        checkpoint_dir="models/checkpoints",
        early_stopping_patience=5
    )
    
    # Guardar modelo y configuración
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), args.model_path)
    with open(args.config_path, "w") as f:
        json.dump(config, f, indent=4)
    
    print(f"\nModelo guardado en: {args.model_path}")
    print(f"Configuración guardada en: {args.config_path}")

def evaluate_model(args):
    """Modo de evaluación."""
    from examples.train_mnist import evaluate_mnist_model
    
    # Cargar configuración del modelo
    with open(args.config_path, "r") as f:
        config = json.load(f)
    
    # Crear modelo
    model = SimpleNN(**config)
    model.load_state_dict(torch.load(args.model_path))
    model = model.to(args.device)
    
    # Cargar datos de prueba
    from examples.train_mnist import load_mnist
    _, _, test_loader, _, _ = load_mnist(batch_size=args.batch_size)
    
    # Evaluar modelo
    test_loss, test_acc = evaluate_mnist_model(
        model=model,
        data_loader=test_loader,
        criterion=torch.nn.CrossEntropyLoss(),
        device=args.device
    )
    
    print(f"\nTest Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.2%}")

def predict(args):
    """Modo de predicción."""
    from torchvision import transforms
    import numpy as np
    
    # Cargar configuración del modelo
    with open(args.config_path, "r") as f:
        config = json.load(f)
    
    # Crear modelo
    model = SimpleNN(**config)
    model.load_state_dict(torch.load(args.model_path))
    model = model.to(args.device)
    model.eval()
    
    # Transformación para las imágenes
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        transforms.Lambda(lambda x: x.view(-1))
    ])
    
    print("\n¡Bienvenido al predictor de MNIST!")
    print("Para salir, presiona Ctrl+C")
    
    while True:
        try:
            # Solicitar al usuario que ingrese el path de la imagen
            image_path = input("\nIntroduce la ruta de una imagen de MNIST: ")
            
            # Cargar y preprocesar la imagen
            from PIL import Image
            image = Image.open(image_path).convert('L')  # Convertir a escala de grises
            image = image.resize((28, 28))  # Redimensionar a 28x28
            input_tensor = transform(image).unsqueeze(0).to(args.device)
            
            # Realizar la predicción
            with torch.no_grad():
                output = model(input_tensor)
                probabilities = torch.nn.functional.softmax(output, dim=1)
                predicted_class = torch.argmax(output, dim=1).item()
                confidence = probabilities[0][predicted_class].item()
            
            # Mostrar resultados
            print(f"\nPredicción: {predicted_class}")
            print(f"Confianza: {confidence:.2%}")
            
        except KeyboardInterrupt:
            print("\n¡Hasta luego!")
            break
        except Exception as e:
            print(f"Error: {str(e)}")

def main():
    """Punto de entrada principal."""
    parser = create_parser()
    args = parser.parse_args()
    
    if args.mode == "train":
        train_model(args)
    elif args.mode == "evaluate":
        evaluate_model(args)
    elif args.mode == "predict":
        predict(args)

if __name__ == "__main__":
    main()
