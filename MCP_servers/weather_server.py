"""MCP Server: Weather
职责：封装天气查询工具，只提供 get_weather。
"""
import logging
import sys

# ⚠️ stdio 模式下不能用 print，必须用 logging 输出到 stderr
logging.basicConfig(level=logging.INFO, stream=sys.stderr)
logger = logging.getLogger("weather_server")

# 让 Python 能找到 tool_agent 根目录的 tools.py
sys.path.insert(0, "E:/tool_agent")

from mcp.server.fastmcp import FastMCP
from tools import get_weather as _get_weather


# 创建 MCP Server，名字叫 "WeatherServer"
mcp = FastMCP("WeatherServer")


@mcp.tool()
def get_weather(city: str) -> str:
    """查询指定城市的当前天气，返回天气状况和温度。"""
    logger.info(f"调用 get_weather: city={city}")
    return _get_weather(city)


if __name__ == "__main__":
    logger.info("WeatherServer 启动，等待 Client 连接...")
    mcp.run(transport="stdio")