#!/usr/bin/env bash
set -euo pipefail

CONTAINER_NAME="${CONTAINER_NAME:-vllm-glm47-flash}"
IMAGE="${VLLM_IMAGE:-vllm/vllm-openai:nightly}"
MODEL_ID="${MODEL_ID:-zai-org/GLM-4.7-Flash}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-glm-4.7-flash}"
HOST_PORT="${HOST_PORT:-8888}"
CONTAINER_PORT="${CONTAINER_PORT:-8888}"
GPU_DEVICE="${GPU_DEVICE:-0}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.82}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-1}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-4096}"
WAIT_TIMEOUT_SECONDS="${WAIT_TIMEOUT_SECONDS:-3600}"

if ! command -v docker >/dev/null 2>&1; then
  echo "docker is required" >&2
  exit 1
fi

if ! command -v curl >/dev/null 2>&1; then
  echo "curl is required" >&2
  exit 1
fi

if command -v nvidia-smi >/dev/null 2>&1; then
  echo "Current GPU memory:"
  nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free \
    --format=csv,noheader,nounits
fi

docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true

docker run -d --name "${CONTAINER_NAME}" \
  --runtime nvidia \
  --gpus "\"device=${GPU_DEVICE}\"" \
  --ipc=host \
  -p "${HOST_PORT}:${CONTAINER_PORT}" \
  -v "${HOME}/.cache/huggingface:/root/.cache/huggingface" \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e VLLM_MLA_DISABLE=1 \
  "${IMAGE}" \
  "${MODEL_ID}" \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --host 0.0.0.0 \
  --port "${CONTAINER_PORT}" \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  --quantization bitsandbytes \
  --kv-cache-dtype fp8 \
  --max-model-len "${MAX_MODEL_LEN}" \
  --max-num-seqs "${MAX_NUM_SEQS}" \
  --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}" \
  --enforce-eager \
  --tool-call-parser glm47 \
  --enable-auto-tool-choice

base_url="http://localhost:${HOST_PORT}"
deadline=$((SECONDS + WAIT_TIMEOUT_SECONDS))

echo "Waiting for ${base_url}/health ..."
until curl --silent --fail --max-time 5 "${base_url}/health" >/dev/null; do
  if ! docker ps --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
    echo "Container exited before becoming healthy. Recent logs:" >&2
    docker logs --tail 200 "${CONTAINER_NAME}" >&2 || true
    exit 1
  fi

  if (( SECONDS >= deadline )); then
    echo "Timed out waiting for vLLM health. Recent logs:" >&2
    docker logs --tail 200 "${CONTAINER_NAME}" >&2 || true
    exit 1
  fi

  sleep 5
done

echo "Checking /v1/models ..."
curl --silent --show-error --fail "${base_url}/v1/models" >/dev/null

echo "Checking /v1/chat/completions ..."
curl --silent --show-error --fail \
  "${base_url}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d "{\"model\":\"${SERVED_MODEL_NAME}\",\"messages\":[{\"role\":\"user\",\"content\":\"Answer with one word: ok?\"}],\"max_tokens\":32,\"temperature\":0}" \
  >/dev/null

echo "vLLM GLM-4.7-Flash is ready at ${base_url}/v1"
