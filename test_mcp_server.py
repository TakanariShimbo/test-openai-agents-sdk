import asyncio
import json
import os
from contextlib import AsyncExitStack
from typing import Any, Literal
from datetime import timedelta

from agents import Agent, Runner, set_default_openai_key
from agents.mcp import (
    MCPServerStdio,
    MCPServerSse,
    MCPServerStreamableHttp,
)
from dotenv import load_dotenv

# ──────────────────────────────────────────────
# Environment
# ──────────────────────────────────────────────
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable is not set")
set_default_openai_key(OPENAI_API_KEY)

# ──────────────────────────────────────────────
# Inline JSON configuration
# ──────────────────────────────────────────────
PARAMS_JSON_STR = r"""
{
  "mcpServers": {
    "playwright": {
      "command": "npx",
      "args": ["-y", "@playwright/mcp@latest"]
    },
    "microsoft.docs.mcp": {
      "type": "stream",
      "url": "https://learn.microsoft.com/api/mcp"
    }
  }
}
"""

# ──────────────────────────────────────────────
# Robust parser
# ──────────────────────────────────────────────
_ALLOWED_TYPES = {"stdio", "sse", "stream", "http", "streamable-http"}
_HTTP_TYPES = {"sse", "stream", "http", "streamable-http"}


def _expand_env(obj: Any) -> Any:
    """Recursively expand environment variables in strings"""
    if isinstance(obj, str):
        return os.path.expandvars(obj)
    if isinstance(obj, list):
        return [_expand_env(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _expand_env(v) for k, v in obj.items()}
    return obj


def parse_mcp_servers_json(json_str: str) -> dict[str, dict[str, Any]]:
    """Parse and validate MCP servers configuration JSON"""
    try:
        root = json.loads(json_str)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON: {exc}") from None

    if not (isinstance(root, dict) and isinstance(root.get("mcpServers"), dict)):
        raise ValueError('Top-level key "mcpServers" must be an object.')

    servers: dict[str, dict[str, Any]] = {}

    for name, cfg in root["mcpServers"].items():
        if not isinstance(name, str):
            raise ValueError("Server names must be strings.")
        if not isinstance(cfg, dict):
            raise ValueError(f'Server "{name}" must be an object.')

        if cfg.get("enabled", True) is False:
            continue

        server_type = cfg.get("type")

        # Validation for stdio vs url
        if "command" in cfg:
            if not isinstance(cfg.get("args"), list):
                raise ValueError(f'Server "{name}": "args" must be a list.')
            if server_type and server_type != "stdio":
                raise ValueError(f'Server "{name}": type must be "stdio" or omitted.')
        elif "url" in cfg:
            if not isinstance(cfg["url"], str):
                raise ValueError(f'Server "{name}": "url" must be a string.')
            if server_type not in _HTTP_TYPES:
                raise ValueError(
                    f'Server "{name}": url servers require type '
                    f'"sse" or "stream"/"http"/"streamable-http".'
                )
        else:
            raise ValueError(
                f'Server "{name}" needs "command" (stdio) or "url" (HTTP/SSE).'
            )

        if server_type and server_type not in _ALLOWED_TYPES:
            raise ValueError(
                f'Server "{name}": unknown type "{server_type}". Allowed: {_ALLOWED_TYPES}'
            )

        # Optional field checks
        if "headers" in cfg and not isinstance(cfg["headers"], dict):
            raise ValueError(f'Server "{name}": "headers" must be an object.')
        if "env" in cfg and not isinstance(cfg["env"], dict):
            raise ValueError(f'Server "{name}": "env" must be an object.')
        if "cwd" in cfg and cfg.get("cwd") is not None and not isinstance(cfg["cwd"], str):
            raise ValueError(f'Server "{name}": "cwd" must be a string.')

        servers[name] = _expand_env(cfg)

    if not servers:
        raise ValueError("No enabled MCP servers found.")
    return servers


# ──────────────────────────────────────────────
# Server builder (no auto-detection)
# ──────────────────────────────────────────────
def build_server(name: str, cfg: dict[str, Any]):
    # ── stdio ────────────────────────────────────────────
    if "command" in cfg:
        stdio_params: dict[str, Any] = {
            "command": cfg["command"],
        }
        if cfg.get("args"):
            stdio_params["args"] = cfg["args"]
        if cfg.get("env"):
            stdio_params["env"] = cfg["env"]
        if cfg.get("cwd"):
            stdio_params["cwd"] = cfg["cwd"]

        return MCPServerStdio(
            name=name,
            params=stdio_params,
            client_session_timeout_seconds=cfg.get("timeout", 30),
        )

    # ── HTTP 系 ──────────────────────────────────────────
    t: Literal["sse", "stream", "http", "streamable-http"] = cfg["type"]

    http_params: dict[str, Any] = {
        "url": cfg["url"],
    }
    if t == "sse":
        if cfg.get("headers"):
            http_params["headers"] = cfg["headers"]
        if cfg.get("timeout"):
            http_params["timeout"] = cfg["timeout"]

        # SSE
        return MCPServerSse(
            name=name,
            params=http_params,
            client_session_timeout_seconds=cfg.get("timeout", 30),
        )
    
    else:
        if cfg.get("headers"):
            http_params["headers"] = cfg["headers"]
        if cfg.get("timeout"):
            http_params["timeout"] = timedelta(seconds=cfg["timeout"])

        # stream / http / streamable-http
        return MCPServerStreamableHttp(
            name=name,
            params=http_params,
            client_session_timeout_seconds=cfg.get("timeout", 30),
        )


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
async def main() -> None:
    """Main execution function"""
    servers_cfg = parse_mcp_servers_json(PARAMS_JSON_STR)

    async with AsyncExitStack() as stack:
        mcp_servers = [
            await stack.enter_async_context(build_server(name, cfg))
            for name, cfg in servers_cfg.items()
        ]
        
        # Warm-up servers by listing tools
        for srv in mcp_servers:
            await srv.list_tools()

        # Create agent
        agent = Agent(
            name="Assistant",
            instructions="あなたは親切なアシスタントです。",
            mcp_servers=mcp_servers,
        )

        # Execute task
        task = (
            "ツールは何がある？"
        )
        result = await Runner.run(agent, task)
        print(result.final_output)

    await asyncio.sleep(1)


if __name__ == "__main__":
    asyncio.run(main())