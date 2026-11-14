#!/bin/bash

echo "========================================"
echo "   FacialPulse - Starting Services"
echo "========================================"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Function to check if port is in use
check_port() {
    lsof -i :$1 > /dev/null 2>&1
    return $?
}

get_port_pid() {
    lsof -ti :$1 2>/dev/null
}

# Cleanup on exit
BACKEND_PID=""
FRONTEND_PID=""

cleanup() {
    echo -e "\n${YELLOW}Shutting down FacialPulse services...${NC}"
    if [ -n "$BACKEND_PID" ]; then
        kill "$BACKEND_PID" 2>/dev/null || true
    fi
    if [ -n "$FRONTEND_PID" ]; then
        kill "$FRONTEND_PID" 2>/dev/null || true
    fi
    exit 0
}

trap cleanup SIGINT SIGTERM

# Optional restart flag: ./start.sh --restart
if [ "$1" == "--restart" ] || [ "$1" == "-r" ]; then
    echo -e "${YELLOW}Killing existing processes on ports 8000 and 3000...${NC}"
    P8000=$(get_port_pid 8000)
    P3000=$(get_port_pid 3000)
    [ -n "$P8000" ] && kill -9 $P8000 2>/dev/null && echo "Freed port 8000 (PID $P8000)"
    [ -n "$P3000" ] && kill -9 $P3000 2>/dev/null && echo "Freed port 3000 (PID $P3000)"
    sleep 1
fi

# Start Backend
echo -e "\n${YELLOW}[1/2] Starting Backend...${NC}"
cd "$PROJECT_DIR/backend"

if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    source venv/bin/activate
    pip install --upgrade pip -q
fi

source venv/bin/activate
python3 -m pip install -r requirements.txt -q

if check_port 8000; then
    echo -e "${YELLOW}Port 8000 already in use (PID: $(get_port_pid 8000)) - using running instance${NC}"
else
    echo "Starting FastAPI server on port 8000..."
    python3 -m uvicorn main:app --host 0.0.0.0 --port 8000 &
    BACKEND_PID=$!
    sleep 3
    if kill -0 "$BACKEND_PID" 2>/dev/null; then
        echo -e "${GREEN}Backend started (PID: $BACKEND_PID)${NC}"
    else
        echo -e "${RED}Backend failed to start. Check logs above.${NC}"
    fi
fi

# Start Frontend
echo -e "\n${YELLOW}[2/2] Starting Frontend...${NC}"
cd "$PROJECT_DIR/frontend"

if [ ! -d "node_modules" ]; then
    echo "Installing npm dependencies..."
    npm install
fi

if check_port 3000; then
    echo -e "${YELLOW}Port 3000 already in use (PID: $(get_port_pid 3000)) - using running instance${NC}"
else
    echo "Starting Next.js server on port 3000..."
    npm run dev &
    FRONTEND_PID=$!
    sleep 4
    if kill -0 "$FRONTEND_PID" 2>/dev/null; then
        echo -e "${GREEN}Frontend started (PID: $FRONTEND_PID)${NC}"
    else
        echo -e "${RED}Frontend failed to start. Check logs above.${NC}"
    fi
fi

echo -e "\n${GREEN}========================================"
echo "   FacialPulse is running!"
echo "========================================"
echo -e "   Frontend: http://localhost:3000"
echo -e "   Backend:  http://localhost:8000"
echo -e "========================================${NC}"
echo ""
echo "Press Ctrl+C to stop all services"

# Wait for user interrupt
wait
