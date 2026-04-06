#!/bin/bash
# Living Mind — Start API Server
# Always uses the venv so all packages (nodriver, etc.) are available

cd "$(dirname "$0")"

if [ ! -f .venv/bin/python ]; then
    echo "ERROR: .venv not found. Run: python3 -m venv .venv && .venv/bin/pip install -r requirements.txt"
    exit 1
fi

exec .venv/bin/python -m api.main "$@"
