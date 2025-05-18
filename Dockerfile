FROM jenkins/jenkins:lts

USER root

# Instalar Docker CLI y dependencias
RUN apt-get update && apt-get install -y \
    docker.io \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Permitir que Jenkins use docker sin sudo (opcional)
RUN usermod -aG docker jenkins

# Instalar plugins necesarios para pipeline y Docker
RUN jenkins-plugin-cli --plugins \
    docker-workflow \
    workflow-aggregator \
    git \
    junit \
    configuration-as-code

USER jenkins

