import asyncio
import json
import os
from contextlib import AsyncExitStack
from typing import Any, Literal, Union, Optional
from datetime import timedelta

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


def validate_server_config(name: Any, cfg: Any) -> dict[str, Any]:
    """個別のサーバー設定を検証"""
    if not isinstance(name, str):
        raise ValueError("Server names must be strings.")
    if not isinstance(cfg, dict):
        raise ValueError(f'Server "{name}" must be an object.')
    return cfg


def validate_server_type_and_connection(name: str, cfg: dict[str, Any]) -> Optional[str]:
    """サーバータイプと接続設定を検証"""
    http_types = {"sse", "stream", "http", "streamable-http"}
    allowed_types = {"stdio"} | http_types
    
    server_type = cfg.get("type")
    
    # stdio（コマンド実行）タイプの検証
    if "command" in cfg:
        if not isinstance(cfg.get("args"), list):
            raise ValueError(f'Server "{name}": "args" must be a list.')
        if server_type and server_type != "stdio":
            raise ValueError(f'Server "{name}": type must be "stdio" or omitted.')
        
    # HTTP/SSEタイプの検証
    elif "url" in cfg:
        if not isinstance(cfg["url"], str):
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


def validate_optional_fields(name: str, cfg: dict[str, Any]) -> None:
    """オプションフィールドを検証"""
    # ヘッダーのチェック
    if "headers" in cfg and not isinstance(cfg["headers"], dict):
        raise ValueError(f'Server "{name}": "headers" must be an object.')
    # 環境変数のチェック
    if "env" in cfg and not isinstance(cfg["env"], dict):
        raise ValueError(f'Server "{name}": "env" must be an object.')
    # 作業ディレクトリのチェック
    if "cwd" in cfg and cfg.get("cwd") is not None and not isinstance(cfg["cwd"], str):
        raise ValueError(f'Server "{name}": "cwd" must be a string.')


def parse_mcp_servers_json(json_str: str) -> dict[str, dict[str, Any]]:
    """MCPサーバー設定のJSONを解析して検証する"""
    root = load_json_configuration(json_str)
    mcp_servers = validate_mcp_servers_structure(root)
    
    servers: dict[str, dict[str, Any]] = {}
    for name, cfg in mcp_servers.items():
        cfg = validate_server_config(name, cfg)
        
        if cfg.get("enabled", True) is False:
            continue
        
        validate_server_type_and_connection(name, cfg)
        validate_optional_fields(name, cfg)
        
        servers[name] = _expand_env(cfg)
    
    if not servers:
        raise ValueError("No enabled MCP servers found.")
    return servers


# ──────────────────────────────────────────────
# Server builder (no auto-detection)
# ──────────────────────────────────────────────

def build_stdio_server(name: str, cfg: dict[str, Any]) -> MCPServerStdio:
    """Stdioタイプのサーバーを構築"""
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


def build_sse_server(name: str, cfg: dict[str, Any]) -> MCPServerSse:
    """SSEタイプのサーバーを構築"""
    http_params: dict[str, Any] = {
        "url": cfg["url"],
    }
    if cfg.get("headers"):
        http_params["headers"] = cfg["headers"]
    
    return MCPServerSse(
        name=name,
        params=http_params,
        client_session_timeout_seconds=cfg.get("timeout", 30),
    )


def build_streamable_http_server(name: str, cfg: dict[str, Any]) -> MCPServerStreamableHttp:
    """StreamableHttpタイプのサーバーを構築"""
    http_params: dict[str, Any] = {
        "url": cfg["url"],
    }
    if cfg.get("headers"):
        http_params["headers"] = cfg["headers"]
    
    return MCPServerStreamableHttp(
        name=name,
        params=http_params,
        client_session_timeout_seconds=cfg.get("timeout", 30),
    )


def build_server(name: str, cfg: dict[str, Any]) -> Union[MCPServerStdio, MCPServerSse, MCPServerStreamableHttp]:
    """設定に基づいて適切なサーバーを構築"""
    if "command" in cfg:
        return build_stdio_server(name, cfg)
    
    server_type: Literal["sse", "stream", "http", "streamable-http"] = cfg["type"]
    
    if server_type == "sse":
        return build_sse_server(name, cfg)
    else:
        return build_streamable_http_server(name, cfg)


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

async def initialize_mcp_servers(
    servers_cfg: dict[str, dict[str, Any]], 
    stack: AsyncExitStack
) -> list[Union[MCPServerStdio, MCPServerSse, MCPServerStreamableHttp]]:
    """MCPサーバーを初期化してリストで返す"""
    mcp_servers = [
        await stack.enter_async_context(build_server(name, cfg))
        for name, cfg in servers_cfg.items()
    ]
    
    # Warm-up servers by listing tools
    for srv in mcp_servers:
        await srv.list_tools()
    
    return mcp_servers


def create_agent(
    mcp_servers: list[Union[MCPServerStdio, MCPServerSse, MCPServerStreamableHttp]]
) -> Agent:
    """エージェントを作成"""
    return Agent(
        name="Assistant",
        instructions="あなたは親切なアシスタントです。",
        mcp_servers=mcp_servers,
    )


async def execute_agent_task(agent: Agent, task: str) -> str:
    """エージェントタスクを実行"""
    result = await Runner.run(agent, task)
    return result.final_output


async def main() -> None:
    """Main execution function"""
    setup_environment()
    
    servers_cfg = parse_mcp_servers_json(PARAMS_JSON_STR)
    
    async with AsyncExitStack() as stack:
        mcp_servers = await initialize_mcp_servers(servers_cfg, stack)
        agent = create_agent(mcp_servers)
        
        task = "ツールは何がある？"
        output = await execute_agent_task(agent, task)
        print(output)
    
    await asyncio.sleep(1)


if __name__ == "__main__":
    asyncio.run(main())