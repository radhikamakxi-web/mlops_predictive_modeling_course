#!/usr/bin/env bash
set -euo pipefail

CONFIG_FILE="${1:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/login.env}"

if [[ ! -f "$CONFIG_FILE" ]]; then
  echo "Docker credentials file not found: $CONFIG_FILE"
  echo "Create one from docker/login.env.example and set DOCKER_USERNAME/DOCKER_PASSWORD."
  exit 1
fi

set -a
source "$CONFIG_FILE"
set +a

if [[ -z "${DOCKER_USERNAME:-}" || -z "${DOCKER_PASSWORD:-}" ]]; then
  echo "DOCKER_USERNAME and DOCKER_PASSWORD must be set in $CONFIG_FILE"
  exit 1
fi

REGISTRY="${DOCKER_REGISTRY:-docker.io}"
echo "$DOCKER_PASSWORD" | docker login "$REGISTRY" -u "$DOCKER_USERNAME" --password-stdin
