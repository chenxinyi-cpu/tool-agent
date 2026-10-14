"""MCP Server: Travel
职责：封装旅游相关工具（景点搜索 + 预算估算）。
"""
import logging
import sys

logging.basicConfig(level=logging.INFO, stream=sys.stderr)
logger = logging.getLogger("travel_server")

sys.path.insert(0, "E:/tool_agent")

from mcp.server.fastmcp import FastMCP
from tools import (
    search_attractions as _search_attractions,
    estimate_budget as _estimate_budget,
)


mcp = FastMCP("TravelServer")


@mcp.tool()
def search_attractions(city: str, days: int = 2) -> str:
    """搜索指定城市的旅游景点推荐，用于制定旅游计划。"""
    logger.info(f"调用 search_attractions: city={city}, days={days}")
    return _search_attractions(city, days)


@mcp.tool()
def estimate_budget(city: str, days: int = 3, level: str = "中等") -> str:
    """搜索指定城市的真实旅游价格（住宿/餐饮/门票），用于估算预算。"""
    logger.info(f"调用 estimate_budget: city={city}, days={days}, level={level}")
    return _estimate_budget(city, days, level)


if __name__ == "__main__":
    logger.info("TravelServer 启动，等待 Client 连接...")
    mcp.run(transport="stdio")