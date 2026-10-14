"""MCP 适配器：把 MCP Server 的工具包装成 Agent 能用的格式。

核心设计：
- 用一个后台 asyncio 线程维持所有 MCP 连接
- Agent（同步）通过 run_coroutine_threadsafe 调用异步 MCP API
- MCP schema 自动转换成 DeepSeek/OpenAI Function Calling 格式
"""
import asyncio
import sys
import threading
from contextlib import AsyncExitStack
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


class MCPToolAdapter:
    def __init__(self, server_configs: dict[str, StdioServerParameters]):
        """
        Args:
            server_configs: {server_name: StdioServerParameters}
        """
        self.server_configs = server_configs
        self.clients: dict[str, ClientSession] = {}   # server_name → session
        self.tool_to_server: dict[str, str] = {}      # tool_name → server_name
        self.openai_tools: list[dict] = []            # DeepSeek/OpenAI 格式的工具列表

        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._stack: AsyncExitStack | None = None

    # ---------- 启动 / 关闭 ----------

    def start(self, timeout: float = 30.0):
        """启动后台 asyncio 线程，连接所有 MCP Server。"""
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

        future = asyncio.run_coroutine_threadsafe(self._connect_all(), self._loop)
        future.result(timeout=timeout)   # 阻塞等待连接完成
        print(f"✅ MCP 适配器启动完成，共 {len(self.openai_tools)} 个工具")

    def _run_loop(self):
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    async def _connect_all(self):
        self._stack = AsyncExitStack()
        for server_name, params in self.server_configs.items():
            try:
                read, write = await self._stack.enter_async_context(stdio_client(params))
                session = await self._stack.enter_async_context(ClientSession(read, write))
                await session.initialize()
                tools = await session.list_tools()

                self.clients[server_name] = session
                for t in tools.tools:
                    self.tool_to_server[t.name] = server_name
                    self.openai_tools.append(self._to_openai_schema(t))

                print(f"  ✅ 已连接 {server_name}，工具: {[t.name for t in tools.tools]}")
            except Exception as e:
                print(f"  ❌ 连接 {server_name} 失败: {e}")

    def stop(self, timeout: float = 10.0):
        """关闭所有连接，停止后台线程。"""
        if self._stack and self._loop:
            try:
                future = asyncio.run_coroutine_threadsafe(self._stack.aclose(), self._loop)
                future.result(timeout=timeout)
            except Exception:
                pass
        if self._loop:
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread:
            self._thread.join(timeout=timeout)
        print("🛑 MCP 适配器已关闭")

    # ---------- 工具调用 ----------

    def call_tool(self, tool_name: str, arguments: dict, timeout: float = 60.0) -> str:
        """同步调用 MCP 工具（内部用异步桥接）。"""
        if tool_name not in self.tool_to_server:
            return f"错误：未找到工具 {tool_name}"

        server_name = self.tool_to_server[tool_name]
        session = self.clients.get(server_name)
        if not session:
            return f"错误：Server {server_name} 未连接"

        try:
            future = asyncio.run_coroutine_threadsafe(
                session.call_tool(tool_name, arguments),
                self._loop,
            )
            result = future.result(timeout=timeout)

            # MCP 返回 result.content 是列表，提取文本
            if hasattr(result, "content") and result.content:
                return "\n".join(
                    c.text for c in result.content
                    if hasattr(c, "text")
                )
            return str(result)
        except Exception as e:
            return f"工具 {tool_name} 执行失败：{e}"

    def has_tool(self, name: str) -> bool:
        return name in self.tool_to_server

    # ---------- Schema 转换 ----------

    @staticmethod
    def _to_openai_schema(mcp_tool) -> dict:
        """把 MCP 工具定义转成 DeepSeek/OpenAI Function Calling 格式。

        MCP: {"name": ..., "description": ..., "inputSchema": {...}}
        OpenAI: {"type": "function", "function": {"name": ..., "description": ..., "parameters": {...}}}
        """
        return {
            "type": "function",
            "function": {
                "name": mcp_tool.name,
                "description": mcp_tool.description or "",
                "parameters": mcp_tool.inputSchema or {
                    "type": "object",
                    "properties": {},
                },
            },
        }


def build_default_adapter() -> MCPToolAdapter:
    """快捷工厂：按默认约定构建适配器（连接 mcp_servers 下的 3 个 Server）。

    关键：显式传递环境变量给子进程，否则 MCP Server 读不到 .env 里的 key。
    """
    import os

    # 从当前进程的环境变量里，挑选要传给子进程的
    # 直接传整个 os.environ，最省事
    child_env = dict(os.environ)

    base = Path(__file__).parent / "mcp_servers"
    configs = {
        "WeatherServer": StdioServerParameters(
            command=sys.executable,
            args=[str(base / "weather_server.py")],
            env=child_env,  # 🆕
        ),
        "TravelServer": StdioServerParameters(
            command=sys.executable,
            args=[str(base / "travel_server.py")],
            env=child_env,  # 🆕
        ),
        "MathServer": StdioServerParameters(
            command=sys.executable,
            args=[str(base / "math_server.py")],
            env=child_env,  # 🆕
        ),
    }
    return MCPToolAdapter(configs)