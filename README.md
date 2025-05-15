# Agente Prueba 1

Un proyecto Python con configuración profesional para desarrollo de alta calidad.

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

Enlace del proyecto: [https://github.com/tu-usuario/agente_prueba1](https://github.com/tu-usuario/agente_prueba1)
