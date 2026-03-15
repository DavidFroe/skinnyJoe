#!/bin/bash
BASE_DIR="$( cd "$( dirname "$(readlink -f "${BASH_SOURCE[0]}")" )" &> /dev/null && pwd )"
exec "$BASE_DIR/venv/bin/python3" "$BASE_DIR/skinnyJoe_cli.py" "$@"
