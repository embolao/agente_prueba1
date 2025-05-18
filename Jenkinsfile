pipeline {
    agent any

    environment {
        DOCKER_IMAGE = 'python:3.12-slim'
    }

    stages {
        stage('Checkout') {
            steps {
                checkout scm
            }
        }

        stage('Linting') {
            steps {
                script {
                    docker.image(env.DOCKER_IMAGE).inside('-u root') {
                        sh '''
                            pip install --user --upgrade pip
                            pip install --user flake8
                            ~/.local/bin/flake8 .
                        '''
                    }
                }
            }
        }

        stage('Testing') {
            steps {
                echo 'Aquí ejecuta tus pruebas (unitarias, integración, etc.)'
                // Ejemplo:
                // sh 'pytest tests/'
            }
        }

        stage('Despliegue Automático') {
            steps {
                echo 'Aquí agrega los pasos para el despliegue automático'
                // Ejemplo:
                // sh './deploy.sh'
            }
        }
    }

    post {
        failure {
            echo '❌ Pipeline falló, revisa los logs para más detalles.'
        }
        success {
            echo '✅ Pipeline ejecutado correctamente.'
        }
        always {
            cleanWs()
        }
    }
}
