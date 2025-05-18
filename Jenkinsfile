pipeline {
    agent {
        docker {
            image 'python:3.12-slim'
            args '--user root'
        }
    }

    stages {
        stage('Preparar Entorno') {
            steps {
                script {
                    echo 'Verificando estructura del proyecto...'
                    sh 'ls -la'
                    echo 'Verificando directorio de tests...'
                    sh 'ls -la tests/'
                    
                    echo 'Instalando dependencias...'
                    sh '''
                        pip install -r requirements.txt
                        pip install pytest pytest-junitxml
                    '''
                }
            }
        }

        stage('Ejecutar Tests') {
            steps {
                script {
                    echo 'Creando directorio para reportes...'
                    sh 'mkdir -p test-reports'
                    
                    echo 'Ejecutando tests...'
                    sh '''
                        pytest --junitxml=test-reports/results.xml -v
                    '''
                }
            }
            post {
                always {
                    echo 'Verificando reporte de tests...'
                    sh 'ls -la test-reports/'
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
            sh 'cat test-reports/results.xml'
        }
    }
}
 
