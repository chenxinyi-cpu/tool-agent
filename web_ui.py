"""Streamlit Web UI for the Agent."""
import streamlit as st
from dotenv import load_dotenv
from llm import LLMClient
from agent import Agent
from memory import Memory

load_dotenv()

SYSTEM_PROMPT = """你是一个智能助手，可以使用工具帮助用户解决问题。

工作原则：
- 需要实时信息（天气、新闻）时调用 get_weather
- 需要计算时调用 calculator
- 如果任何工具返回了英文内容，请用中文总结给用户
- 用户要求规划旅游时，按以下步骤进行：
  1. 用 get_weather 查询目的地天气
  2. 用 search_attractions 搜索景点推荐
  3. 用 estimate_budget 搜索真实价格
  4. 根据真实价格用 calculator 计算总预算
  5. 输出清晰的行程安排和预算明细
  """

@st.cache_resource
def get_agent():
    """Agent 实例只在第一次调用时创建，之后复用。"""
    llm=LLMClient(model="deepseek-chat")
    memory=Memory(path="./memory.json")
    return Agent(
        llm=llm,
        system_prompt=SYSTEM_PROMPT,
        memory=memory,
        max_iterations=8,
        context_limit=8000,
    )

def main():
    st.set_page_config(page_title="小陈Agent",page_icon="🤖")
    st.title("🤖 小陈 Agent")
    st.caption("支持工具调用 + 长期记忆")

    agent = get_agent()

    # ==================== 侧边栏 ====================
    with st.sidebar:
        st.header("⚙️ 控制面板")
        if st.button("🗑️ 清空当前对话", use_container_width=True):
            st.session_state.messages = []
            agent.history = []  # 也清掉 Agent 的内部历史
            agent.summary = ""
            st.rerun()
        st.divider()

        # 显示当前记忆
        st.subheader("🧠 长期记忆")
        if agent.memory and agent.memory.facts:
            st.caption(f"共 {len(agent.memory.facts)} 条")
            for fact in agent.memory.facts[-10:]:  # 只显示最近 10 条
                st.write(f"- {fact['content']}")
        else:
            st.caption("暂无记忆")

        st.divider()

        # 显示状态
        st.subheader("📊 状态")
        st.write(f"当前历史消息数：{len(agent.history)}")
        st.write(f"上下文压缩状态：{'已压缩' if agent.summary else '未压缩'}")

    # ==================== 主区：对话 ====================
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    if user_input := st.chat_input("问点什么..."):   #边赋值边判断
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.write(user_input)
        with st.chat_message("assistant"):
            with st.spinner("思考中..."):
                answer=agent.run(user_input)
            st.write(answer)

        st.session_state.messages.append({"role": "assistant", "content": answer})

        # 每次对话后刷新侧边栏（显示最新记忆）
        st.rerun()

if __name__ == "__main__":
    main()




















