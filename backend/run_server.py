"""One-click runner for FastAPI 3DGS Studio Backend Server."""

import os
import sys
from pathlib import Path

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))

import uvicorn

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    host = os.environ.get("HOST", "0.0.0.0")
    print(f"Starting 3DGS Generation API backend on http://localhost:{port}")
    uvicorn.run("backend.app:app", host=host, port=port, reload=False)
