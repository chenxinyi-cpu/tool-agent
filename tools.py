import ast
import operator
from http.client import responses
from skill_loader import load_skill as _load_skill_impl
from rag.retriever import search_knowledge_base

import requests

def get_weather(city:str)->str:
    try:
        url=f"https://wttr.in/{city}?format=j1"
        response=requests.get(url)
        response.raise_for_status()
        data=response.json()
        cond=data["current_condition"][0]
        desc=cond["weatherDesc"][0]["value"]
        temp=cond["temp_C"]
        return f"{city}当前天气：{desc}，温度：{temp}°C"
    except Exception as e:
        return f"查询天气失败：{e}"

def calculator(expression: str) -> str:
    """安全的数学计算（用 ast 而不是 eval，防止代码注入）"""
    allowed = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Pow: operator.pow,
        ast.USub: operator.neg,
    }
    def _eval(node):
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.BinOp):
            op = allowed.get(type(node.op))
            if not op:
                raise ValueError(f"不支持的运算符：{type(node.op)}")
            return op(_eval(node.left), _eval(node.right))
        if isinstance(node, ast.UnaryOp):
            op = allowed.get(type(node.op))
            if not op:
                raise ValueError(f"不支持的一元运算符：{type(node.op)}")
            return op(_eval(node.operand))
        raise ValueError(f"不支持的表达式类型：{type(node)}")

    try:
        expr = expression.replace("×", "*").replace("÷", "/")
        tree = ast.parse(expr, mode="eval")
        result = _eval(tree.body)
        if isinstance(result, float) and result == int(result):
            result = int(result)
        return str(result)
    except Exception as e:
        return f"计算错误：{e}"

def search_attractions(city:str,days:int=3)->str:
    """用 Tavily 搜索指定城市的旅游景点推荐"""
    import os
    api_key=os.environ.get("TAVILY_API_KEY")
    if not api_key:
        return "未配置 TAVILY_API_KEY，无法搜索景点"

    try:
        from tavily import TavilyClient
        client = TavilyClient(api_key=api_key)
        query=f"{city}{days}天旅游必去景点推荐 行程安排"
        response=client.search(
            query=query,
            search_depth="basic",
            include_answer=True,
            max_results=5,
        )
        if response.get("answer"):
            return response["answer"]

        results=response.get("results",[])
        if not results:
            return f"没有找到{city}的景点信息"
        lines=[f"-{r['title']}:{r['content'][:150]}" for r in results]
        return "\n".join(lines)
    except Exception as e:
        return f"景点信息搜索失败{e}"


def estimate_budget(city:str,days:int=3,level:str="中等")->str:
    """用 Tavily 搜索真实价格数据，返回住宿/餐饮/门票的参考价格。
       注意：本工具只返回搜索到的真实价格，不直接计算总和——让 calculator 算。
    """
    import os
    api_key=os.environ.get("TAVILY_API_KEY")
    if not api_key:
        return "未配置 TAVILY_API_KEY，无法搜索真实价格"

    try:
        from tavily import TavilyClient
        client = TavilyClient(api_key=api_key)
        # 分三类搜真实价格
        queries = {
            "住宿": f"{city} {level}酒店 每晚价格 多少钱",
            "餐饮": f"{city} 人均餐饮消费 每天多少钱",
            "景点门票": f"{city} 主要景点 门票价格 多少钱",
        }
        results={}
        for category,query in queries.items():
            response = client.search(
                query=query,
                search_depth="basic",
                include_answer=True,
                max_results=3,
            )
            if response.get("answer"):
                results[category] = response["answer"]
            else:
                snippets = [r["content"][:200] for r in response.get("results", [])[:2]]
                results[category] = "\n".join(snippets) if snippets else "未找到相关价格"
        output=[f"# {city} 真实价格参考（搜索自网络）\n"]
        for category,result in results.items():
            output.append(f"##{category}: {result}\n")
        output.append(
            f"\n提示：以上是 {city} {days} 天 {level} 档次的真实价格数据。"
            f"请根据这些数据，用 calculator 计算 {days} 天的总预算（不含往返机票）。"
        )
        return "\n".join(output)

    except Exception as e:
        return f"搜索价格失败：{e}"





TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "查询指定城市的当前天气",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "城市名，如 '北京'"}
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "执行数学计算，支持 + - * / ** 和括号",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string","description": "数学表达式，如 '(12+34)*5'",
                    }
                },
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_attractions",
            "description":"搜索指定城市的旅游景点推荐，用于制定旅游计划",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "城市名，如 '成都'"},
                    "days": {"type": "integer", "description": "旅游天数，默认 3"}
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "estimate_budget",
            "description": "搜索指定城市的真实旅游价格（住宿/餐饮/门票），用于估算预算",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "城市名"},
                    "days": {"type": "integer", "description": "旅游天数"},
                    "level": {
                        "type": "string",
                        "enum": ["经济", "中等", "高端"],
                        "description": "消费档次，默认'中等'"
                    }
                },
                "required": ["city", "days"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "load_skill",
            "description": "加载一个 skill 的完整流程说明。当任务匹配某个 skill 时调用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "skill 名字，例如 'medical'",
                    }
                },
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_knowledge_base",
            "description": (
                "从本地知识库检索相关文档片段。"
                "适用于：医疗健康类问题（症状、用药、就医建议）、"
                "询问本项目的实现细节（MCP、Skill、记忆机制等）。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "检索关键词或自然语言问题",
                    },
                    "k": {
                        "type": "integer",
                        "description": "返回文档数量，默认 3",
                    },
                },
                "required": ["query"],
            },
        },
    },
]

TOOL_REGISTRY = {
    "get_weather": get_weather,
    "calculator": calculator,
    "search_attractions": search_attractions,
    "estimate_budget": estimate_budget,
    "load_skill": _load_skill_impl,
    "search_knowledge_base": search_knowledge_base,
}


def execute_tool(tool_name:str, tool_args:dict) -> str:
    if tool_name not in TOOL_REGISTRY:
        return f"未找到{tool_name}工具"
    try:
        return TOOL_REGISTRY[tool_name](**tool_args)
    except Exception as e:
        return f"调用{tool_name}工具失败{e}"











