#!/bin/zsh
set -euo pipefail

if [[ -z "${HF_TOKEN:-}" && -z "${HUGGINGFACE_HUB_TOKEN:-}" ]]; then
  echo "Set HF_TOKEN or HUGGINGFACE_HUB_TOKEN first."
  exit 1
fi

python3 app.py
