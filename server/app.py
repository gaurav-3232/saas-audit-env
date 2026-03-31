"""Server entry point for OpenEnv multi-mode deployment."""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.main import app


def main(host="0.0.0.0", port=7860):
    """Entry point for direct execution via uv run or python -m."""
    import asyncio
    from hypercorn.config import Config
    from hypercorn.asyncio import serve

    hconfig = Config()
    hconfig.bind = [f"{host}:{port}"]
    hconfig.workers = 1
    asyncio.run(serve(app, hconfig))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    args = parser.parse_args()
    main(host=args.host, port=args.port)
