pipeline {
    agent {
        docker {
            image 'python:3.12-slim'
        }
    }

    stages {
        stage('Ejecutar Tests') {
            steps {
                sh '''
                    set -e
                    pip install -r requirements.txt
                    pip install pytest

                    # Añadir src/ al PYTHONPATH
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
            echo '¡Los tests se completaron con éxito!'
        }
        failure {
            echo 'Los tests fallaron. Revisando logs detallados...'
            sh 'cat test-reports/results.xml || echo "No se pudo leer el archivo de resultados."'
        }
    }
}
