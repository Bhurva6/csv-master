#!/bin/bash

# Activate virtual environment and run Streamlit app
echo "🚀 Starting CSV Data Explorer..."
echo "📊 Your app will open at: http://localhost:8501"
echo ""

# Check if virtual environment exists
if [ -d ".venv" ]; then
    source .venv/bin/activate
    streamlit run app.py
else
    echo "❌ Virtual environment not found!"
    echo "Please run: python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt"
fi
