#!/bin/bash
# Start the OpenAI-compatible API server

cd "$(dirname "$0")"
source venv/bin/activate

echo "=" | head -c 70 && echo
echo "Starting Nemotron Next 8B PEFT API Server"
echo "=" | head -c 70 && echo

# Check if server is already running
if curl -s http://localhost:8000/ > /dev/null 2>&1; then
    echo "⚠️  Server is already running on port 8000"
    echo "   Stop it first or use a different port:"
    echo "   python serve_model.py --port 8001"
    exit 1
fi

# Start server
echo "Starting server on http://0.0.0.0:8000"
echo "Press Ctrl+C to stop"
echo ""

python serve_model.py --port 8000 --host 0.0.0.0

