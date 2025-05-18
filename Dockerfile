FROM jenkins/jenkins:lts

USER root

# Instalar Docker CLI y dependencias necesarias
RUN apt-get update && apt-get install -y \
    docker.io \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Añadir el usuario jenkins al grupo docker para poder usar docker sin sudo
RUN usermod -aG docker jenkins

# Instalar plugins necesarios para pipeline, docker y git
RUN jenkins-plugin-cli --plugins \
    docker-workflow \
    workflow-aggregator \
    git \
    junit \
    configuration-as-code

USER jenkins


