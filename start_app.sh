#!/bin/bash

# Start Flask application for PDFKG

echo "========================================="
echo "PDFKG Flask Application Starter"
echo "========================================="

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "❌ Virtual environment not found!"
    echo "Please create it first: python -m venv .venv"
    exit 1
fi

# Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate

# Check if Flask dependencies are installed
if ! python -c "import flask" 2>/dev/null; then
    echo "❌ Flask not installed!"
    echo "Installing dependencies from requirements.txt..."
    pip install -r requirements.txt
fi

# Check if Docker services are running
echo "Checking Docker services..."
if ! docker ps | grep -q "arangodb\|milvus"; then
    echo "⚠️  Docker services not running!"
    echo "Starting ArangoDB and Milvus..."
    docker-compose up -d
    echo "Waiting for services to start..."
    sleep 5
fi

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo "⚠️  .env file not found! Creating from .env.example..."
    if [ -f ".env.example" ]; then
        cp .env.example .env
        echo "✅ Created .env file. Please configure your API keys!"
    else
        echo "❌ .env.example not found!"
        exit 1
    fi
fi

echo ""
echo "========================================="
echo "Starting Flask application..."
echo "========================================="
echo ""

# Run Flask application
python app.py
