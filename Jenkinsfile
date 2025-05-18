pipeline {
    agent {
        docker {
            image 'python:3.12-slim'
        }
    }

    stages {
        stage('Checkout') {
            steps {
                checkout scm
            }
        }

        stage('Ejecutar Tests') {
            steps {
                sh 'apt-get update && apt-get install -y gcc' // si lo necesitas
                sh 'pip install -r requirements.txt'
                sh 'mkdir -p test-reports'
                sh 'pytest -v --junitxml=test-reports/results.xml'
            }
            post {
                always {
                    junit 'test-reports/*.xml'
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
            echo 'Los tests fallaron. Por favor, revisa los logs.'
        }
    }
}
