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
                    pip install -r requirements.txt
                    pip install pytest pytest-junitxml
                    # Asegurar que se pueda importar el módulo
                    export PYTHONPATH=$PYTHONPATH:$(pwd)
                    
                    # Verifica presencia del módulo
                    ls -la
                    ls -la agente_prueba1 || echo "El módulo no está presente"

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

 
