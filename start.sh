
#!/usr/bin/env bash

HOST=127.0.0.1

PIDS=()

start_server() {
    local PORT=$1
    local CMD=$2

    if curl -fs "http://$HOST:$PORT/v1/models" >/dev/null; then
        echo "✓ Server on port $PORT already running."
        return
    fi

    echo "Starting server on port $PORT..."

    eval "$CMD" &

    PIDS+=($!)

    until curl -fs "http://$HOST:$PORT/v1/models" >/dev/null; do
        sleep 1
    done

    echo "✓ Server on port $PORT ready."
}

start_server \
    8081 \
    'CUDA_VISIBLE_DEVICES="" ~/apps/llama.cpp/build/bin/llama-server \
        -m ~/models/Qwen3-Embedding-0.6B-Q8_0.gguf \
        --embeddings \
        --port 8081'

start_server \
    8080 \
    '~/apps/llama.cpp/build/bin/llama-server \
        -m ~/models/Qwen3-4B-Q5_K_M.gguf \
        --port 8080 \
        -ngl 99'

venv/bin/python main.py

# Закрываем только то, что сами запустили
for PID in "${PIDS[@]}"; do
    kill "$PID"
    wait "$PID" 2>/dev/null
done

