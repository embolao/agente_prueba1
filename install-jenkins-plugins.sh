#!/bin/bash

# Lista de plugins a instalar
PLUGINS="pipeline-github-lib workflow-aggregator docker-workflow blueocean credentials-binding"

# Instalar cada plugin
for PLUGIN in $PLUGINS; do
  echo "Instalando $PLUGIN..."
  docker exec jenkins jenkins-plugin-cli --plugin "$PLUGIN"
done

echo "Reiniciando Jenkins..."
docker restart jenkins

echo "Esperando a que Jenkins se reinicie..."
sleep 30

echo "Plugins instalados y Jenkins reiniciado."
