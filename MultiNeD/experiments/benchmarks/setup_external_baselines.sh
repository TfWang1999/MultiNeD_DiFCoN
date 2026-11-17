#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PROJECT_DIR="$(cd "${CODE_DIR}/.." && pwd)"
EXTERNAL_ROOT="${EXTERNAL_BASELINE_ROOT:-$(cd "${PROJECT_DIR}/.." && pwd)/external_baselines}"
mkdir -p "${EXTERNAL_ROOT}"

MIOFLOW_COMMIT="36365403d0f23cc3ad1065781c7331bf81debf4e"
TIGON_COMMIT="838c17dce2687147dbc51e36bd837c31d4dc5d20"

if [[ ! -d "${EXTERNAL_ROOT}/MIOFlow/.git" ]]; then
  git clone https://github.com/KrishnaswamyLab/MIOFlow.git "${EXTERNAL_ROOT}/MIOFlow"
fi
git -C "${EXTERNAL_ROOT}/MIOFlow" fetch origin "${MIOFLOW_COMMIT}"
git -C "${EXTERNAL_ROOT}/MIOFlow" checkout --detach "${MIOFLOW_COMMIT}"

if [[ ! -d "${EXTERNAL_ROOT}/TIGON/.git" ]]; then
  git clone https://github.com/yutongo/TIGON.git "${EXTERNAL_ROOT}/TIGON"
fi
git -C "${EXTERNAL_ROOT}/TIGON" fetch origin "${TIGON_COMMIT}"
git -C "${EXTERNAL_ROOT}/TIGON" checkout --detach "${TIGON_COMMIT}"

if ! conda run -n multined_mioflow python -c "import sys" >/dev/null 2>&1; then
  conda create -y -n multined_mioflow python=3.11
fi
conda run -n multined_mioflow python -m pip install --upgrade pip
conda run -n multined_mioflow python -m pip install -e "${EXTERNAL_ROOT}/MIOFlow"

if ! conda run -n multined_tigon python -c "import sys" >/dev/null 2>&1; then
  conda create -y -n multined_tigon python=3.10
fi
conda run -n multined_tigon python -m pip install --upgrade pip
conda run -n multined_tigon python -m pip install \
  torch==1.13.1 numpy==1.23.5 scipy==1.10.1 \
  torchdiffeq==0.2.3 TorchDiffEqPack==1.0.1 \
  matplotlib==3.5.3 seaborn==0.12.2

printf 'External baseline root: %s\n' "${EXTERNAL_ROOT}"
printf 'MIOFLOW_REPO=%s\n' "${EXTERNAL_ROOT}/MIOFlow"
printf 'TIGON_REPO=%s\n' "${EXTERNAL_ROOT}/TIGON"
