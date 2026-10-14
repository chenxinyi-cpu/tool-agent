"""Agent Eval：自动跑测试用例，统计工具调用准确率。

用法：
    cd E:\tool_agent
    python eval/run_eval.py
"""
import json
import sys
import time
from pathlib import Path

# 让 Python 能找到 tool_agent 根目录的模块
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
from llm import LLMClient
from agent import Agent
from memory import Memory
from mcp_adapter import build_default_adapter
load_dotenv()


# ============ 拦截工具调用，记录调用了哪些工具 ============
class InstrumentedAdapter:
    """包装 MCP 适配器，记录每次工具调用。"""
    def __init__(self, adapter):
        self._adapter = adapter
        self.called_tools = []   # 记录本次任务调用了哪些工具

    def __getattr__(self, name):
        return getattr(self._adapter, name)

    def call_tool(self, tool_name, arguments, timeout=60.0):
        self.called_tools.append(tool_name)
        return self._adapter.call_tool(tool_name, arguments, timeout=timeout)

    def has_tool(self, name):
        return self._adapter.has_tool(name)

    @property
    def openai_tools(self):
        return self._adapter.openai_tools


# ============ 评测单个用例 ============
def run_case(agent, instrumented_adapter, case):
    """跑一个测试用例，返回结果。"""
    instrumented_adapter.called_tools = []   # 清空

    start = time.time()
    try:
        answer = agent.run(case["task"])
        error = None
    except Exception as e:
        answer = ""
        error = str(e)
    latency = time.time() - start

    called = set(instrumented_adapter.called_tools)
    expected = set(case["expected_tools"])

    # 判断标准：
    # 1. 精确匹配：called == expected
    # 2. 包含匹配：expected 中的工具都被调用了（called 可以多）
    exact_match = called == expected
    contains_match = expected.issubset(called)

    return {
        "id": case["id"],
        "task": case["task"],
        "expected": list(expected),
        "called": list(called),
        "exact_match": exact_match,
        "contains_match": contains_match,
        "latency": latency,
        "error": error,
        "answer_preview": answer[:80] if answer else "",
    }


def main():
    print("=" * 70)
    print("Agent Eval - 开始评测")
    print("=" * 70)

    # 加载测试用例
    cases_file = Path(__file__).parent / "test_cases.json"
    with open(cases_file, "r", encoding="utf-8") as f:
        cases = json.load(f)
    print(f"加载了 {len(cases)} 个测试用例\n")

    # 启动 MCP 适配器
    print("启动 MCP 适配器...")
    raw_adapter = build_default_adapter()
    raw_adapter.start()
    adapter = InstrumentedAdapter(raw_adapter)
    print()

    # 创建 Agent（记忆用临时文件，避免污染）
    llm = LLMClient(model="deepseek-chat")
    memory = Memory(path="eval/eval_memory.json", llm=llm)
    agent = Agent(
        llm=llm,
        system_prompt="你是一个智能助手，可以使用工具帮助用户解决问题。",
        memory=memory,
        max_iterations=8,
        context_limit=8000,
        mcp_adapter=adapter,
    )

    # 逐个跑测试用例
    results = []
    for i, case in enumerate(cases, 1):
        print(f"[{i}/{len(cases)}] 任务: {case['task']}")
        result = run_case(agent, adapter, case)
        results.append(result)

        status = "✅" if result["exact_match"] else ("⚠️" if result["contains_match"] else "❌")
        print(f"    {status} 期望: {result['expected']}")
        print(f"       实际: {result['called']}")
        print(f"       耗时: {result['latency']:.1f}s")
        if result["error"]:
            print(f"       错误: {result['error']}")
        print()

    # ============ 生成报告 ============
    print("=" * 70)
    print("评测报告")
    print("=" * 70)

    total = len(results)
    exact_pass = sum(1 for r in results if r["exact_match"])
    contains_pass = sum(1 for r in results if r["contains_match"])
    errors = sum(1 for r in results if r["error"])
    avg_latency = sum(r["latency"] for r in results) / total

    print(f"\n总用例数: {total}")
    print(f"精确匹配: {exact_pass}/{total}  ({exact_pass/total*100:.1f}%)")
    print(f"包含匹配: {contains_pass}/{total}  ({contains_pass/total*100:.1f}%)")
    print(f"报错数:   {errors}/{total}")
    print(f"平均延迟: {avg_latency:.2f}s")

    # 按类别统计
    print("\n按类别统计:")
    for category in ["single_tool", "multi_tool", "no_tool"]:
        cat_results = [r for r in results if cases[r["id"]-1]["category"] == category]
        if not cat_results:
            continue
        cat_pass = sum(1 for r in cat_results if r["contains_match"])
        print(f"  {category}: {cat_pass}/{len(cat_results)}")

    # 失败用例
    failed = [r for r in results if not r["contains_match"]]
    if failed:
        print("\n失败用例:")
        for r in failed:
            print(f"  [{r['id']}] {r['task']}")
            print(f"       期望: {r['expected']}")
            print(f"       实际: {r['called']}")

    # 保存报告
    report_file = Path(__file__).parent / "eval_report.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump({
            "total": total,
            "exact_pass": exact_pass,
            "contains_pass": contains_pass,
            "errors": errors,
            "avg_latency": avg_latency,
            "results": results,
        }, f, ensure_ascii=False, indent=2)
    print(f"\n📄 详细报告已保存到: {report_file}")

    # 关闭
    raw_adapter.stop()


if __name__ == "__main__":
    main()