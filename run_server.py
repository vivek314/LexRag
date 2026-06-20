# run_server.py — Convenient server entry point for LexRAG UI

import uvicorn
import yaml

if __name__ == "__main__":
    # Load configuration to get port/host settings
    with open("configs/config.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    
    host = cfg["api"]["host"]
    port = cfg["api"]["port"]
    
    print(f"============================================================")
    print(f"Launching LexRAG Web UI sandbox on: http://{host}:{port}")
    print(f"Press CTRL+C to shut down the server.")
    print(f"============================================================")
    
    uvicorn.run("src.api.main:app", host=host, port=port, reload=True)
