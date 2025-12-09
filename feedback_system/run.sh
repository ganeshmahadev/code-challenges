#!/bin/bash
# Unified run script for the feedback analysis system
#
# Usage:
#   ./run.sh              - Start interactive CLI (default)
#   ./run.sh server       - Start API server
#   ./run.sh test         - Run unified test suite

MODE=${1:-interactive}

echo "Customer Feedback Analysis System"
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo " Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -q -r requirements.txt

# Check for .env file
if [ ! -f ".env" ]; then
    echo "Creating .env from template..."
    cp .env.example .env
    echo "WARNING: Using default configuration. Edit .env to add your OPENAI_API_KEY"
    echo ""
fi

# Run based on mode
case "$MODE" in
    interactive)
        echo " Starting Interactive CLI Mode"
        echo ""
        echo "Press Ctrl+C to exit"
        echo ""
        python3 interactive_cli.py
        ;;

    server)
        echo " Starting API server on http://localhost:8000"
        echo ""
        echo "API Endpoints:"
        echo "  - POST /feedback (requires header: X-API-Key: test-api-key-12345)"
        echo "  - GET  /health"
        echo ""
        echo "Press Ctrl+C to stop"
        echo ""
        uvicorn main:app --reload --port 8000
        ;;

    test)
        echo " Running Tests"
        echo ""
        pytest test_system.py -v --tb=short
        ;;

    *)
        echo " Unknown mode: $MODE"
        echo ""
        echo "Usage:"
        echo "  ./run.sh              - Start interactive CLI (default)"
        echo "  ./run.sh server       - Start API server"
        echo "  ./run.sh test         - Run unified test suite"
        exit 1
        ;;
esac
