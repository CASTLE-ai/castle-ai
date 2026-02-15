"""MCP server CLI command."""

import typer

app = typer.Typer(help="MCP (Model Context Protocol) server")


@app.command("start")
def start(
    transport: str = typer.Option(
        "stdio", help="Transport type: stdio or http"
    ),
    port: int = typer.Option(8000, help="HTTP port (only for http transport)"),
):
    """Start the CASTLE MCP server."""
    from castle.mcp.server import mcp as _mcp

    if transport == "stdio":
        _mcp.run(transport="stdio")
    elif transport == "http":
        # Port is set via FastMCP settings; override before run
        _mcp.settings.port = port  # type: ignore[attr-defined]
        _mcp.run(transport="streamable-http")
    else:
        typer.echo(f"Unknown transport: {transport}", err=True)
        raise typer.Exit(1)
