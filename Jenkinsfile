pipeline {
    agent {
        docker {
            image 'python:3.12-slim'
        }
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

        stage('Ejecutar Tests') {
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
    }

    post {
        always {
            cleanWs()
        }
        success {
            echo '✅ Pipeline completado con éxito.'
        }
        failure {
            echo '❌ Algo falló. Revisa los logs.'
            sh 'cat test-reports/results.xml || echo "No se pudo leer el archivo de resultados."'
        }
    }
}
