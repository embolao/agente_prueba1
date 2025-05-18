# Agente Prueba 1

Proyecto de clasificación de imágenes MNIST usando redes neuronales con PyTorch.

## 🚀 Características

- 🎨 Interfaz de línea de comandos para entrenamiento y predicción
- 🔧 Configuración flexible de hiperparámetros
- 📊 Visualización de métricas de entrenamiento
- 📈 Soporte para GPU/CUDA
- 🧪 Pruebas unitarias completas
- 🛠️ Integración con pre-commit hooks
- 📦 Configuración de Docker para desarrollo y producción
- 🤖 GitHub Actions para CI/CD
- 📚 Documentación completa

## 📋 Requisitos

- Python 3.8+
- PyTorch 2.7.0+
- CUDA (opcional, para aceleración GPU)
- Docker y Docker Compose (opcional, para desarrollo con contenedores)
- Git

## 🛠️ Instalación

1. **Clonar el repositorio**:
   ```bash
git clone https://github.com/embolao/agente_prueba1.git
cd agente_prueba1
```

2. **Configurar entorno virtual**:
   ```bash
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
```

3. **Instalar dependencias**:
   ```bash
pip install -e .
```

4. **Instalar pre-commit hooks**:
   ```bash
pre-commit install
```

## 📊 Uso de la aplicación

La aplicación ofrece tres modos principales de uso:

### 1. Entrenamiento de un nuevo modelo

```bash
python app.py train [opciones]
```

Opciones disponibles:
- `--epochs`: Número de épocas de entrenamiento (default: 10)
- `--batch-size`: Tamaño del batch (default: 64)
- `--learning-rate`: Tasa de aprendizaje (default: 0.001)
- `--device`: Dispositivo para el entrenamiento (cpu o cuda) (default: cuda si disponible)
- `--hidden-sizes`: Tamaños de las capas ocultas (default: [256, 128, 64])
- `--dropout`: Probabilidad de dropout (default: 0.2)
- `--activation`: Función de activación (relu, sigmoid, tanh) (default: relu)

Ejemplo de uso:
```bash
python app.py train --epochs 10 --batch-size 128 --learning-rate 0.0001
```

### 2. Evaluación de un modelo

```bash
python app.py evaluate [opciones]
```

Opciones disponibles:
- `--device`: Dispositivo para la evaluación (cpu o cuda)
- `--model-path`: Ruta del modelo a cargar
- `--config-path`: Ruta de la configuración del modelo
- `--batch-size`: Tamaño del batch

Ejemplo de uso:
```bash
python app.py evaluate --model-path models/mnist_model.pth
```

### 3. Predicción con un modelo

```bash
python app.py predict [opciones]
```

Opciones disponibles:
- `--device`: Dispositivo para la predicción (cpu o cuda)
- `--model-path`: Ruta del modelo a cargar
- `--config-path`: Ruta de la configuración del modelo

Ejemplo de uso:
```bash
python app.py predict --model-path models/mnist_model.pth
```

## 🧪 Ejecución de pruebas

```bash
pytest
```

Para ver el informe de cobertura:
```bash
pytest --cov=src --cov-report=term-missing
```

## 🧹 Formateo y Linting

- **Formatear código**:
  ```bash
  black .
  ```

- **Verificar estilo de código**:
  ```bash
  flake8 .
  ```

- **Verificar tipos**:
  ```bash
  mypy .
  ```

## 📊 Métricas del modelo

El modelo utiliza las siguientes métricas:

- **Pérdida**: CrossEntropyLoss
- **Precisión**: Accuracy
- **Tasa de aprendizaje**: AdamW
- **Regularización**:
  - Weight decay: 1e-4
  - Dropout: Configurable
  - Batch Normalization: Enabled
## 🤝 Contribución
1. Haz un fork del proyecto
2. Crea una rama para tu característica (`git checkout -b feature/AmazingFeature`)
3. Haz commit de tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. Haz push a la rama (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

## 📄 Licencia

Distribuido bajo la licencia MIT. Ver `LICENSE` para más información.

## 📚 Documentación adicional

- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [MNIST Dataset](http://yann.lecun.com/exdb/mnist/)
- [TensorBoard Documentation](https://www.tensorflow.org/tensorboard)

## 🚀 Características

- Formateo de código con Black
- Linting con Flake8
- Verificación de tipos con MyPy
- Pruebas unitarias con Pytest
- Integración con pre-commit hooks
- Configuración de Docker para desarrollo y producción
- GitHub Actions para CI/CD

## 🛠️ Configuración del entorno

### Requisitos previos

- Python 3.8+
- Docker y Docker Compose (opcional, para desarrollo con contenedores)
- Git

### Configuración del entorno de desarrollo

1. **Clonar el repositorio**:
   ```bash
   git clone https://github.com/tu-usuario/agente_prueba1.git
   cd agente_prueba1
   ```

2. **Configurar entorno virtual (opcional pero recomendado)**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # En Windows: venv\Scripts\activate
   ```

3. **Instalar dependencias**:
   ```bash
   pip install -e .[dev]
   ```

4. **Instalar pre-commit hooks**:
   ```bash
   pre-commit install
   ```

### Usando Docker (opcional)

1. **Construir y ejecutar el contenedor de desarrollo**:
   ```bash
   docker-compose up -d app
   docker-compose exec app bash
   ```

2. **Ejecutar pruebas**:
   ```bash
   docker-compose run --rm tests
   ```

3. **Ejecutar linters**:
   ```bash
   docker-compose run --rm lint
   ```

## 🧪 Ejecutando pruebas

```bash
pytest
```

Para ver el informe de cobertura:

```bash
pytest --cov=src --cov-report=term-missing
```

## 🧹 Formateo y Linting

- **Formatear código**:
  ```bash
  black .
  ```

- **Verificar estilo de código**:
  ```bash
  flake8 .
  ```

- **Verificar tipos**:
  ```bash
  mypy .
  ```

## 🚀 Despliegue

1. **Construir la imagen de producción**:
   ```bash
   docker build --target production -t agente-prueba1 .
   ```

2. **Ejecutar el contenedor de producción**:
   ```bash
   docker run -p 8000:8000 agente-prueba1
   ```

## 🤝 Contribución

1. Haz un fork del proyecto
2. Crea una rama para tu característica (`git checkout -b feature/AmazingFeature`)
3. Haz commit de tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. Haz push a la rama (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

## 📄 Licencia

Distribuido bajo la licencia MIT. Ver `LICENSE` para más información.

## 🏗️ Integración Continua con Jenkins

Este proyecto incluye un `Jenkinsfile` configurado para ejecutar una canalización de CI/CD que incluye:

1. **Preparación del entorno**: Instalación de dependencias
2. **Formateo de código**: Verificación con Black e isort
3. **Análisis estático**: Linting con Flake8 y verificación de tipos con MyPy
4. **Ejecución de pruebas**: Pruebas unitarias con cobertura
5. **Construcción del paquete**: Creación de paquetes fuente y wheel

### Configuración en Jenkins

1. **Instalar plugins necesarios**:
   - Docker Pipeline
   - Pipeline
   - JUnit
   - Cobertura
   - Blue Ocean (opcional, para mejor visualización)

2. **Crear un nuevo ítem** en Jenkins:
   - Seleccionar "Pipeline"
   - Especificar la URL del repositorio Git
   - En la configuración de la pipeline, seleccionar "Pipeline script from SCM"
   - Especificar la rama principal (ej: `main` o `master`)
   - Guardar la configuración

3. **Ejecutar la pipeline** manualmente o configurar un webhook para ejecución automática en cada push.

### Variables de entorno recomendadas

Configura las siguientes variables de entorno en Jenkins para un mejor rendimiento:

- `PYTHONUNBUFFERED=1`: Para ver la salida de Python en tiempo real
- `PYTHONDONTWRITEBYTECODE=1`: Para evitar archivos `.pyc`
- `PIP_DISABLE_PIP_VERSION_CHECK=on`: Para acelerar la instalación de paquetes

## ✉️ Contacto
Embolao

Enlace del proyecto: [https://github.com/embolao/agente_prueba1](https://github.com/embolao/agente_prueba1)
