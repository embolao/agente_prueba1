pipeline {
    agent {
        docker {
            image 'python:3.12-slim'
            args '--user root'
        }
    }

    environment {
        PYTHONUNBUFFERED = '1'
        PYTHONDONTWRITEBYTECODE = '1'
        PIP_DISABLE_PIP_VERSION_CHECK = 'on'
    }

    stages {
        stage('Preparar entorno') {
            steps {
                sh 'python --version'
                sh 'python -m pip install --upgrade pip'
                sh 'python -m pip install -e .[dev]'
            }
        }

        stage('Formatear código') {
            steps {
                sh 'black --check .'
                sh 'isort --check-only .'
            }
        }

        stage('Análisis estático') {
            steps {
                sh 'flake8 .'
                sh 'mypy .'
            }
        }

        stage('Ejecutar pruebas') {
            steps {
                sh 'pytest --cov=src --cov-report=xml:coverage.xml -v'
            }
            post {
                always {
                    junit '**/test-reports/*.xml'
                    cobertura(coberturaReportFile: '**/coverage.xml')
                }
            }
        }

        stage('Construir paquete') {
            steps {
                sh 'python setup.py sdist bdist_wheel'
                archiveArtifacts 'dist/*'
            }
        }
    }

    post {
        always {
            // Limpiar el workspace después de la ejecución
            cleanWs()
        }
        success {
            echo '¡Todas las etapas se completaron con éxito!'
        }
        failure {
            echo 'La canalización ha fallado. Por favor, revisa los logs.'
        }
    }
}
