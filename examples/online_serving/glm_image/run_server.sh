#!/bin/bash
# GLM-Image online serving startup script

MODEL="${MODEL:-zai-org/GLM-Image}"
PORT="${PORT:-8000}"

echo "Starting GLM-Image server..."
echo "Model: $MODEL"
echo "Port: $PORT"

vllm serve "$MODEL" --omni \
    --port "$PORT"
