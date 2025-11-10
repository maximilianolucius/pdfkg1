#!/usr/bin/env python3
"""
Flask web application for PDF Knowledge Graph Q&A.

Migration from Gradio to Flask with Socket.IO for real-time chat.

Usage:
    python flask_app.py
"""

# Fix for macOS multiprocessing segmentation fault
import multiprocessing
import sys
if sys.platform == "darwin":  # macOS
    try:
        multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

import os

# Fix FAISS threading issues on macOS
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

from flask import Flask, render_template
from flask_socketio import SocketIO
from flask_cors import CORS
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import Flask app components
from flask_app.config import Config
from flask_app.system_logger import system_logger
from flask_app.storage_init import init_storage, verify_services

# Create Flask app
app = Flask(__name__,
            template_folder='flask_app/templates',
            static_folder='flask_app/static')
app.config.from_object(Config)

# Enable CORS for API endpoints
CORS(app, resources={r"/api/*": {"origins": "*"}})

# Initialize Socket.IO with async mode
socketio = SocketIO(app,
                   cors_allowed_origins="*",
                   async_mode='threading',
                   logger=True,
                   engineio_logger=False)

# ==========================================
# STARTUP VERIFICATION & INITIALIZATION
# ==========================================

system_logger.log("=" * 80, "INFO")
system_logger.log("PDFKG Flask Application Starting...", "INFO")
system_logger.log("=" * 80, "INFO")
system_logger.log("Verifying database services...", "INFO")

# Verify ArangoDB and Milvus are running
services_ok, error_message = verify_services()

if not services_ok:
    system_logger.log("=" * 80, "ERROR")
    system_logger.log("CRITICAL: Required services are not available!", "ERROR")
    system_logger.log(error_message, "ERROR")
    system_logger.log("=" * 80, "ERROR")
    system_logger.log("Please ensure ArangoDB and Milvus are running:", "ERROR")
    system_logger.log("  docker-compose up -d", "ERROR")
    system_logger.log("=" * 80, "ERROR")
    print("\n" + system_logger.get_logs())
    sys.exit(1)

# Initialize storage backend
storage = init_storage()
app.config['storage'] = storage

# ==========================================
# REGISTER BLUEPRINTS
# ==========================================

from flask_app.routes.main import main_bp
from flask_app.routes.api import api_bp

app.register_blueprint(main_bp)
app.register_blueprint(api_bp, url_prefix='/api')

# ==========================================
# SOCKET.IO EVENT HANDLERS
# ==========================================

from flask_app import socketio_handlers
socketio_handlers.register_handlers(socketio, storage)

# ==========================================
# ERROR HANDLERS
# ==========================================

@app.errorhandler(404)
def not_found(error):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(error):
    system_logger.log(f"Internal Server Error: {error}", "ERROR")
    return render_template('500.html'), 500

# ==========================================
# MAIN ENTRY POINT
# ==========================================

if __name__ == "__main__":
    print("=" * 80)
    print("PDF Knowledge Graph Q&A - Flask Web App")
    print("=" * 80)

    # Check LLM providers
    llm_status = []
    if os.getenv("GEMINI_API_KEY"):
        gemini_model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        llm_status.append(f"✅ Gemini enabled: {gemini_model}")
    else:
        llm_status.append("⚠️  Gemini disabled: GEMINI_API_KEY not set in .env")

    if os.getenv("MISTRAL_API_KEY"):
        mistral_model = os.getenv("MISTRAL_MODEL", "mistral-large-latest")
        llm_status.append(f"✅ Mistral enabled: {mistral_model}")
    else:
        llm_status.append("⚠️  Mistral disabled: MISTRAL_API_KEY not set in .env")

    for status in llm_status:
        print(status)

    print("=" * 80)
    print("\n🚀 Starting Flask server with Socket.IO...\n")

    # Try ports in sequence until one is available
    ports_to_try = [8016, 8017, 8018, 8019, 7860]
    server_port = None

    for port in ports_to_try:
        try:
            socketio.run(app,
                        host="0.0.0.0",
                        port=port,
                        debug=False,
                        use_reloader=False,
                        allow_unsafe_werkzeug=True)
            server_port = port
            break  # Success!
        except OSError as e:
            if "Address already in use" in str(e):
                print(f"⚠️  Port {port} is busy, trying next port...")
                continue
            else:
                raise  # Re-raise if it's a different error

    if server_port is None:
        print(f"\n❌ ERROR: All ports {ports_to_try} are busy!")
        print("Kill existing processes with: pkill -f 'python.*flask_app.py'")
        sys.exit(1)
