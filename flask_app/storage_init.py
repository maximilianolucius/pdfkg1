"""Storage backend initialization and service verification."""
import os
from flask_app.system_logger import system_logger
from pdfkg.storage import get_storage_backend, FileStorage


def verify_services():
    """
    Verify that Milvus and ArangoDB are running and accessible.
    Returns (success: bool, error_message: str)
    """
    errors = []

    # Check ArangoDB
    try:
        from arango import ArangoClient
        arango_host = os.getenv("ARANGO_HOST", "localhost")
        arango_port = os.getenv("ARANGO_PORT", "8529")
        arango_user = os.getenv("ARANGO_USER", "root")
        arango_password = os.getenv("ARANGO_PASSWORD", "")

        client = ArangoClient(hosts=f"http://{arango_host}:{arango_port}")
        sys_db = client.db('_system', username=arango_user, password=arango_password)
        version = sys_db.version()
        system_logger.log(f"✅ ArangoDB connected: version {version}", "INFO")
    except Exception as e:
        error_msg = f"❌ ArangoDB connection failed: {e}"
        system_logger.log(error_msg, "ERROR")
        errors.append(error_msg)

    # Check Milvus
    try:
        from pymilvus import connections, utility
        milvus_host = os.getenv("MILVUS_HOST", "localhost")
        milvus_port = os.getenv("MILVUS_PORT", "19530")

        connections.connect(
            alias="verify",
            host=milvus_host,
            port=milvus_port
        )

        # Try to list collections as a connection test
        collections = utility.list_collections(using="verify")
        connections.disconnect("verify")
        system_logger.log(f"✅ Milvus connected: {len(collections)} collections found", "INFO")
    except Exception as e:
        error_msg = f"❌ Milvus connection failed: {e}"
        system_logger.log(error_msg, "ERROR")
        errors.append(error_msg)

    if errors:
        return False, "\n".join(errors)

    return True, ""


def init_storage():
    """Initialize storage backend with error handling."""
    try:
        storage = get_storage_backend()
        storage_type = os.getenv("STORAGE_BACKEND", "arango")
        system_logger.log(f"✅ Storage backend initialized: {storage_type}", "INFO")
        return storage
    except Exception as e:
        system_logger.log(f"⚠️  Storage backend initialization failed: {e}", "ERROR")
        system_logger.log(f"⚠️  Using file storage as fallback", "WARNING")
        storage = FileStorage()
        return storage
