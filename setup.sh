#!/bin/bash

# Exit on error
set -e

# Configuration
ENTERPRISE_PORT=8000
OPENSOURCE_PORT=8001
FRONTEND_PORT=8502

# Function to create and activate virtual environment
setup_venv() {
    local dir=$1
    echo "Setting up virtual environment in $dir"
    cd "$dir"
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
}

# Function to check if port is in use
check_port() {
    local port=$1
    if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null ; then
        echo "Port $port is already in use. Attempting to free it..."
        sudo kill -9 $(lsof -t -i:$port) || true
        sleep 2
    fi
}

# Function to start a service
start_service() {
    local service=$1
    local port=$2
    echo "Starting $service service on port $port"
    cd "backend/${service}_service"
    setup_venv .
    
    if [ "$service" = "enterprise" ]; then
        # Enterprise-specific environment variables
        export PDF_SERVICES_CLIENT_ID=${PDF_SERVICES_CLIENT_ID:-"YOUR_CLIENT_ID"}
        export PDF_SERVICES_CLIENT_SECRET=${PDF_SERVICES_CLIENT_SECRET:-"YOUR_CLIENT_SECRET"}
    fi
    
    # Common AWS credentials
    export AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID:-"YOUR_ACCESS_KEY"}
    export AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY:-"YOUR_SECRET_KEY"}
    
    # Check and clear port if needed
    check_port $port
    
    echo "Starting uvicorn for $service service on port $port"
    uvicorn main:app --reload --port $port &
    sleep 5  # Wait for service to start
}

main() {
    # Install system dependencies
    echo "Updating system and installing dependencies..."
    sudo apt update
    sudo apt install -y python3 python3-pip
    
    # Store initial directory
    BASE_DIR=$(pwd)
    
    # Start backend services with different ports
    start_service "enterprise" $ENTERPRISE_PORT
    cd "$BASE_DIR"
    start_service "opensource" $OPENSOURCE_PORT
    
    # Check and clear frontend port if needed
    check_port $FRONTEND_PORT
    
    # Start frontend
    echo "Starting frontend on port $FRONTEND_PORT..."
    cd "$BASE_DIR/frontend"
    setup_venv .
    streamlit run streamlit_app.py --server.port $FRONTEND_PORT
}

# Cleanup function
cleanup() {
    echo "Cleaning up processes..."
    pkill -f "uvicorn.*:$ENTERPRISE_PORT" || true
    pkill -f "uvicorn.*:$OPENSOURCE_PORT" || true
    pkill -f "streamlit.*:$FRONTEND_PORT" || true
}

# Set up trap for cleanup on script exit
trap cleanup EXIT

main