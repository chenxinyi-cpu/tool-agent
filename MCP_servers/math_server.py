"""MCP Server: Math
职责：封装通用数学计算工具，任何 Agent 都能复用。
"""
import logging
import sys

logging.basicConfig(level=logging.INFO, stream=sys.stderr)
logger = logging.getLogger("math_server")

sys.path.insert(0, "E:/tool_agent")

from mcp.server.fastmcp import FastMCP
from tools import calculator as _calculator


mcp = FastMCP("MathServer")


@mcp.tool()
def calculator(expression: str) -> str:
    """执行数学计算，支持 + - * / ** 和括号。"""
    logger.info(f"调用 calculator: {expression}")
    return _calculator(expression)


if __name__ == "__main__":
    logger.info("MathServer 启动，等待 Client 连接...")
    mcp.run(transport="stdio")