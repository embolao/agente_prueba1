pipeline {
    agent {
        docker {
            image 'python:3.12-slim'
        }
    }

    stages {
        stage('Ejecutar Tests') {
            steps {
                sh 'pip install -r requirements.txt'
                sh 'pytest -v'
            }
            post {
                always {
                    junit '**/test-reports/*.xml'
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
