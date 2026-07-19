import argparse
import sys
import uvicorn
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        prog="minieval-pro",
        description="MiniEval Pro — LLM hallucination detection dashboard"
    )

    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to run dashboard on (default: 8000)"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host to bind to (default: 127.0.0.1)"
    )
    parser.add_argument(
        "--version",
        action="store_true",
        help="Show version"
    )
    parser.add_argument(
        "command",
        nargs="?",
        choices=["init", "version"],
        help="Command to run"
    )

    args = parser.parse_args()

    if args.version or args.command == "version":
        print("minieval-pro v1.0.0")
        sys.exit(0)

    if args.command == "init":
        print("Initializing MiniEval Pro database...")
        try:
            from minieval_pro.web.app import init_db
            init_db()
            print("Database initialized.")
            print("Run 'minieval-pro' to start the dashboard.")
        except Exception as e:
            print(f"Init failed: {e}")
            sys.exit(1)
        return

    # Default — start dashboard
    print("=" * 45)
    print("  MiniEval Pro — LLM Hallucination Detection")
    print("=" * 45)
    print(f"  Dashboard: http://{args.host}:{args.port}")
    print("  Press Ctrl+C to stop")
    print("=" * 45)

    try:
        uvicorn.run(
            "minieval_pro.web.app:app",
            host=args.host,
            port=args.port,
            reload=False
        )
    except KeyboardInterrupt:
        print("\nStopped.")