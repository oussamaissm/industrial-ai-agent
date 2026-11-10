import argparse

from src.backends import get_backend, resolve_backend_name
from src.mcp_server.server import create_server


def main() -> None:
    p = argparse.ArgumentParser(description="Industrial factory MCP server (read-only)")
    p.add_argument("--backend", default=None, help="simulator | openfactory (default: $FACTORY_BACKEND or simulator)")
    p.add_argument("--transport", default="stdio", choices=["stdio", "streamable-http", "sse"])
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    args = p.parse_args()

    server = create_server(get_backend(resolve_backend_name(args.backend)))
    if args.transport == "stdio":
        server.run("stdio")
    else:
        server.run(args.transport, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
