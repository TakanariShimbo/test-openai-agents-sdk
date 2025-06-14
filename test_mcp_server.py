import asyncio
import json
import os
from contextlib import AsyncExitStack
from typing import Any, Literal, Union, Optional

from agents import Agent, Runner, set_default_openai_key
from agents.mcp import (
    MCPServerStdio,
    MCPServerSse,
    MCPServerStreamableHttp,
)
from dotenv import load_dotenv

# ──────────────────────────────────────────────
# 環境設定
# ──────────────────────────────────────────────
def setup_environment() -> None:
    """環境変数の設定とOpenAI APIキーのセットアップ"""
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable is not set")
    set_default_openai_key(api_key)

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
def _expand_env(obj: Any) -> Any:
    """環境変数を展開する"""
    if isinstance(obj, str):
        return os.path.expandvars(obj)
    if isinstance(obj, list):
        return [_expand_env(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _expand_env(v) for k, v in obj.items()}
    return obj


def load_json_configuration(json_str: str) -> dict[str, Any]:
    """JSON文字列をパースして設定を取得"""
    try:
        return json.loads(json_str)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON: {exc}") from None


def validate_mcp_servers_structure(root: Any) -> dict[str, Any]:
    """MCPサーバー設定の基本構造を検証"""
    if not (isinstance(root, dict) and isinstance(root.get("mcpServers"), dict)):
        raise ValueError('Top-level key "mcpServers" must be an object.')
    return root["mcpServers"]


def validate_server_params(name: Any, params: Any) -> dict[str, Any]:
    """個別のサーバー設定を検証"""
    if not isinstance(name, str):
        raise ValueError("Server names must be strings.")
    if not isinstance(params, dict):
        raise ValueError(f'Server "{name}" must be an object.')
    return params


def validate_server_type_and_connection(name: str, params: dict[str, Any]) -> Optional[str]:
    """サーバータイプと接続設定を検証"""
    http_types = {"sse", "stream", "http", "streamable-http"}
    allowed_types = {"stdio"} | http_types

    server_type = params.get("type")
    
    # stdio（コマンド実行）タイプの検証
    if "command" in params:
        # argsは省略可能だが、指定する場合はリストである必要がある
        args = params.get("args")
        if args is not None and not isinstance(args, list):
            raise ValueError(f'Server "{name}": "args" must be a list.')
        if server_type and server_type != "stdio":
            raise ValueError(f'Server "{name}": type must be "stdio" or omitted.')
        
    # HTTP/SSEタイプの検証
    elif "url" in params:
        if not isinstance(params["url"], str):
            raise ValueError(f'Server "{name}": "url" must be a string.')
        if server_type not in http_types:
            raise ValueError(
                f'Server "{name}": url servers require type '
                f'"sse" or "stream"/"http"/"streamable-http".'
            )
    else:
        raise ValueError(
            f'Server "{name}" needs "command" (stdio) or "url" (HTTP/SSE).'
        )
    
    if server_type and server_type not in allowed_types:
        raise ValueError(
            f'Server "{name}": unknown type "{server_type}". Allowed: {allowed_types}'
        )
    
    return server_type


def validate_optional_fields(name: str, params: dict[str, Any]) -> None:
    """オプションフィールドを検証"""
    # ヘッダーのチェック
    if "headers" in params and not isinstance(params["headers"], dict):
        raise ValueError(f'Server "{name}": "headers" must be an object.')
    # 環境変数のチェック
    if "env" in params and not isinstance(params["env"], dict):
        raise ValueError(f'Server "{name}": "env" must be an object.')
    # 作業ディレクトリのチェック
    if "cwd" in params and params.get("cwd") is not None and not isinstance(params["cwd"], str):
        raise ValueError(f'Server "{name}": "cwd" must be a string.')


def parse_mcp_servers_json(json_str: str) -> dict[str, dict[str, Any]]:
    """MCPサーバー設定のJSONを解析して検証する"""
    root = load_json_configuration(json_str)
    mcp_servers = validate_mcp_servers_structure(root)
    
    servers: dict[str, dict[str, Any]] = {}
    for name, params in mcp_servers.items():
        params = validate_server_params(name, params)
        
        if params.get("enabled", True) is False:
            continue
        
        validate_server_type_and_connection(name, params)
        validate_optional_fields(name, params)
        
        servers[name] = _expand_env(params)
    
    if not servers:
        raise ValueError("No enabled MCP servers found.")
    return servers


# ──────────────────────────────────────────────
# Server builder (no auto-detection)
# ──────────────────────────────────────────────

def build_stdio_server(name: str, params: dict[str, Any]) -> MCPServerStdio:
    """Stdioタイプのサーバーを構築"""
    stdio_params: dict[str, Any] = {
        "command": params["command"],
    }
    if params.get("args"):
        stdio_params["args"] = params["args"]
    if params.get("env"):
        stdio_params["env"] = params["env"]
    if params.get("cwd"):
        stdio_params["cwd"] = params["cwd"]
    
    return MCPServerStdio(
        name=name,
        params=stdio_params,
        client_session_timeout_seconds=params.get("timeout", 30),
    )


def build_sse_server(name: str, params: dict[str, Any]) -> MCPServerSse:
    """SSEタイプのサーバーを構築"""
    http_params: dict[str, Any] = {
        "url": params["url"],
    }
    if params.get("headers"):
        http_params["headers"] = params["headers"]
    
    return MCPServerSse(
        name=name,
        params=http_params,
        client_session_timeout_seconds=params.get("timeout", 30),
    )


def build_streamable_http_server(name: str, params: dict[str, Any]) -> MCPServerStreamableHttp:
    """StreamableHttpタイプのサーバーを構築"""
    http_params: dict[str, Any] = {
        "url": params["url"],
    }
    if params.get("headers"):
        http_params["headers"] = params["headers"]
    
    return MCPServerStreamableHttp(
        name=name,
        params=http_params,
        client_session_timeout_seconds=params.get("timeout", 30),
    )


def build_server(name: str, params: dict[str, Any]) -> Union[MCPServerStdio, MCPServerSse, MCPServerStreamableHttp]:
    """設定に基づいて適切なサーバーを構築"""
    if "command" in params:
        return build_stdio_server(name, params)
    
    server_type: Literal["sse", "stream", "http", "streamable-http"] = params["type"]
    
    if server_type == "sse":
        return build_sse_server(name, params)
    else:
        return build_streamable_http_server(name, params)


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

async def main() -> None:
    """メイン実行関数"""
    setup_environment()
    
    servers_params = parse_mcp_servers_json(PARAMS_JSON_STR)
    
    async with AsyncExitStack() as stack:
        mcp_servers = []
        for name, params in servers_params.items():
            mcp_server = build_server(name, params)
            await stack.enter_async_context(mcp_server)
            mcp_servers.append(mcp_server)

        agent = Agent(
            name="Assistant",
            instructions="あなたは親切なアシスタントです。",
            mcp_servers=mcp_servers,
        )
        
        task = "ツールは何がある？"
        result = await Runner.run(agent, task)
        print(result.final_output)
    
    await asyncio.sleep(1)


if __name__ == "__main__":
    asyncio.run(main())