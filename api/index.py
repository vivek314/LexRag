# Vercel Python serverless entrypoint.
# Vercel discovers the `app` FastAPI/ASGI object and serves it.
import sys
from pathlib import Path

# Ensure project root is on the Python path so `src.*` imports resolve.
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.api.main import app  # noqa: F401
