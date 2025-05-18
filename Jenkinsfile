pipeline {
    agent any

    environment {
        COMPOSE_PROJECT_NAME = 'mi_app'
    }

    stages {
        stage('Linting') {
            agent {
                docker {
                    image 'python:3.12-slim'
                }
            }
            steps {
                sh '''
                    pip install flake8
                    flake8 src/ --exit-zero --statistics
                '''
            }
        }

        stage('Testing') {
            agent {
                docker {
                    image 'python:3.12-slim'
                }
            }
            steps {
                sh '''
                    pip install -r requirements.txt pytest
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
                    set -e
                    docker-compose down
                    docker-compose up -d --build
                    docker-compose ps
                    sleep 10
                    curl --fail http://localhost:8080/
                '''
            }
        }
    }

    post {
        always {
            cleanWs()
        }
        success {
            echo '✅ Pipeline completo: Linting, Tests y Despliegue OK.'
        }
        failure {
            echo '❌ Error en pipeline, revisa logs.'
        }
    }
}
