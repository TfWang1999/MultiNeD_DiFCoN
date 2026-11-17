#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PROJECT_DIR="$(cd "${CODE_DIR}/.." && pwd)"
EXTERNAL_ROOT="${EXTERNAL_BASELINE_ROOT:-$(cd "${PROJECT_DIR}/.." && pwd)/external_baselines}"
OTCFM_REPO="${OTCFM_REPO:-${EXTERNAL_ROOT}/conditional-flow-matching}"
OTCFM_TAG="${OTCFM_TAG:-1.0.7}"
OTCFM_ENV="${OTCFM_ENV:-multined_otcfm}"
OTCFM_TORCH_VERSION="${OTCFM_TORCH_VERSION:-2.2.2}"
OTCFM_TORCH_INDEX_URL="${OTCFM_TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu121}"

mkdir -p "${EXTERNAL_ROOT}"
if [[ ! -d "${OTCFM_REPO}/.git" ]]; then
  git clone https://github.com/atong01/conditional-flow-matching.git "${OTCFM_REPO}"
fi
git -C "${OTCFM_REPO}" fetch --tags origin
git -C "${OTCFM_REPO}" checkout --detach "${OTCFM_TAG}"

if ! conda run -n "${OTCFM_ENV}" python -c "import sys" >/dev/null 2>&1; then
  conda create -y -n "${OTCFM_ENV}" python=3.11
fi
conda run -n "${OTCFM_ENV}" python -m pip install --upgrade pip
conda run -n "${OTCFM_ENV}" python -m pip install \
  "torch==${OTCFM_TORCH_VERSION}" --index-url "${OTCFM_TORCH_INDEX_URL}"
conda run -n "${OTCFM_ENV}" python -m pip install "numpy<2"
conda run -n "${OTCFM_ENV}" python -m pip install -e "${OTCFM_REPO}"
conda run -n "${OTCFM_ENV}" python -c \
  "import torch, torchcfm; from torchcfm.conditional_flow_matching import ExactOptimalTransportConditionalFlowMatcher; print('torch', torch.__version__, 'torchcfm', torchcfm.__version__, ExactOptimalTransportConditionalFlowMatcher.__name__)"

CONDA_BASE="$(conda info --base)"
printf 'OTCFM_REPO=%s\n' "${OTCFM_REPO}"
printf 'OTCFM_PYTHON=%s\n' "${CONDA_BASE}/envs/${OTCFM_ENV}/bin/python"
