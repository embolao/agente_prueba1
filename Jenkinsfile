pipeline {
    agent {
        docker {
            image 'python:3.12-slim'
        }
    }

    environment {
        COMPOSE_PROJECT_NAME = 'mi_app'
    }

    stages {
        stage('Linting') {
            steps {
                sh '''
                    pip install flake8
                    echo "Ejecutando flake8 en src/..."
                    flake8 src/ --exit-zero --statistics
                '''
            }
        }

        stage('Testing') {
            steps {
                sh '''
                    set -e
                    pip install -r requirements.txt
                    pip install pytest

                    export PYTHONPATH=$PYTHONPATH:$(pwd)/src

                    mkdir -p test-reports
                    pytest --junitxml=test-reports/results.xml -v
                '''
            }
            post {
                always {
                    junit 'test-reports/results.xml'
                }
            }
        }

        stage('Despliegue Automático') {
            when {
                branch 'master'
            }
            steps {
                sh '''
                    echo "Iniciando despliegue con docker-compose..."
                    docker-compose down
                    docker-compose up -d --build
                    docker-compose ps

                    echo "Esperando 10 segundos para que arranque el contenedor..."
                    sleep 10

                    echo "Verificando estado del servicio..."
                    curl --fail http://localhost:8080/ || {
                        echo "❌ La aplicación no respondió correctamente.";
                        exit 1;
                    }

                    echo "✅ Verificación de salud completada."
                '''
            }
        }
    }

    post {
        always {
            cleanWs()
        }
        success {
            echo '✅ Pipeline completo: Linting, Tests, Despliegue y Verificación exitosos.'
        }
        failure {
            echo '❌ Fallo en alguna etapa. Revisa los errores.'
        }
    }
}
