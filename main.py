"""入口：多轮对话循环。"""
from dotenv import load_dotenv
from llm import LLMClient
from agent import Agent
from memory import Memory
from skill_loader import format_skill_list_for_prompt

load_dotenv()

SYSTEM_PROMPT = """你是一个智能助手，可以使用工具帮助用户解决问题。

工作原则：
- 需要实时信息（天气、新闻）时调用 get_weather
- 需要计算时调用 calculator
- **如果任何工具返回了英文内容，请用中文总结给用户**
- 用户要求规划旅游时，按以下步骤进行：
  1. 用 get_weather 查询目的地天气（决定户外/室内活动）
  2. 用 search_attractions 搜索景点推荐
  3. 用 estimate_budget 搜索真实价格（住宿/餐饮/门票）
  4. 根据搜索到的真实价格，用 calculator 计算总预算
  5. 综合以上信息，输出一份清晰的行程安排（按 Day 1 / Day 2 / Day 3 分天），
     并在末尾给出预算明细（包含每一项价格和总价）

得到足够信息后，用自然语言给出完整、清晰的回答。"""+ "\n\n" + format_skill_list_for_prompt()

def main():
    llm = LLMClient(model="deepseek-chat")
    memory = Memory(path="memory.json",llm=llm)
    agent = Agent(
        llm=llm,
        system_prompt=SYSTEM_PROMPT,
        memory=memory,
        max_iterations=8,
        context_limit=8000,
    )

    print("=" * 50)
    print("Agent 已启动。输入 'exit' 退出。")
    print("=" * 50)

    while True:
        try:
            user_input = input("\n你: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit", "退出"):
            break

        # 流式模式：/stream 你的问题
        if user_input.startswith("/stream "):
            question = user_input[8:].strip()
            print("\n助手: ", end="", flush=True)
            for chunk in agent.run_stream(question):
                print(chunk, end="", flush=True)
            print()  # 末尾换行
            continue

        print()
        answer = agent.run(user_input)
        print(f"\n助手: {answer}")


if __name__ == "__main__":
    main()