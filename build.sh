#!/usr/bin/env bash
#python3.13 -m venv venv # if does not exist
#venv/bin/pip install requirments.txt

PORT=8081
HOST=127.0.0.1

# if need install llama.cpp
# Проверяем, работает ли сервер
if curl -fs "http://$HOST:$PORT/v1/models" >/dev/null; then
    echo "llama-server already running."
    STARTED_BY_SCRIPT=0
else
    echo "Starting llama-server..."

    CUDA_VISIBLE_DEVICES="" \
    ~/apps/llama.cpp/build/bin/llama-server \
        -m ~/models/Qwen3-Embedding-0.6B-Q8_0.gguf \
        --embeddings \
        --port $PORT &

    LLAMA_PID=$!
    STARTED_BY_SCRIPT=1

    echo "Waiting for server..."

    until curl -fs "http://$HOST:$PORT/v1/models" >/dev/null; do
        sleep 1
    done
fi

venv/bin/python build.py

# Завершаем только тот сервер, который сами запустили
if [ "$STARTED_BY_SCRIPT" -eq 1 ]; then
    kill "$LLAMA_PID"
    wait "$LLAMA_PID" 2>/dev/null
fi




