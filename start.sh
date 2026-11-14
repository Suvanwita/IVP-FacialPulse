#!/bin/bash
# Root launcher forwarding to 09_Source_Code/start.sh
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
exec "$SCRIPT_DIR/09_Source_Code/start.sh" "$@"
