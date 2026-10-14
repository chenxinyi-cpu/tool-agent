"""Agent 主循环：ReAct + 上下文压缩 + 记忆注入/提取。"""
import json
from llm import LLMClient
from memory import Memory
from tools import TOOLS_SCHEMA, execute_tool

class Agent:
    def __init__(self, llm:LLMClient,system_prompt:str,
                 memory:Memory=None,max_iterations:int=3,context_limit:int=4000,
                 mcp_adapter=None):
        self.llm = llm
        self.system_prompt = system_prompt
        self.memory = memory
        self.max_iterations = max_iterations
        self.context_limit = context_limit
        self.history = []  # 对话历史（user / assistant / tool 消息）
        self.summary = ""  # 上下文压缩后的历史摘要

        self.mcp_adapter = mcp_adapter
        if mcp_adapter is not None:
            self.tools_schema = mcp_adapter.openai_tools
        else:
            from tools import TOOLS_SCHEMA
            self.tools_schema = TOOLS_SCHEMA

    def run(self,user_input:str)->str:
        messages = self._build_messages(user_input)
        for step in range(self.max_iterations):
            msg = self.llm.chat(messages, tools=self.tools_schema)

            if not msg.tool_calls:
                answer=msg.content or ""
                self.history.append({"role":"user","content":user_input})
                self.history.append({"role":"assistant","content":answer})
                self._extract_memories(user_input, answer)
                self._maybe_compress()
                return answer

            messages.append({
                "role":"assistant",
                "content":msg.content or "",
                "tool_calls":[
                    {
                        "id":tc.id,
                        "type":"function",
                        "function":{
                            "name":tc.function.name,
                            "arguments":tc.function.arguments,
                        },
                    }
                    for tc in msg.tool_calls
                ],
            })
            for tc in msg.tool_calls:
                args=json.loads(tc.function.arguments)
                print(f"调用工具{tc.function.name}（{args}）")
                if self.mcp_adapter and self.mcp_adapter.has_tool(tc.function.name):
                    result = self.mcp_adapter.call_tool(tc.function.name, args)
                else:
                    from tools import execute_tool
                    result = execute_tool(tc.function.name, args)
                print(f"{result[:100]}")
                messages.append({
                    "role":"tool",
                    "tool_call_id":tc.id,
                    "content":result,
                })
        return "抱歉，未在规定步骤完成任务"

    # ---------- 流式运行（不支持工具调用，适合简单问答） ----------
    def run_stream(self,user_input:str):
        """流式运行，逐块 yield 文本片段。调用者负责打印和拼接。"""
        messages = self._build_messages(user_input)
        full_answer = ""
        for chunk in self.llm.chat_stream(messages):
            full_answer += chunk
            yield chunk
        # 流式结束后，保存历史 + 提取记忆（和 run 一样）
        self.history.append({"role": "user", "content": user_input})
        self.history.append({"role": "assistant", "content": full_answer})
        self._extract_memories(user_input, full_answer)
        self._maybe_compress()


    def _build_messages(self, user_input:str)->list:
        system=self.system_prompt
        if self.summary:
            system+=f"\n\n##之前对话的摘要{self.summary}"
        if self.memory:
            facts=self.memory.recall(user_input,k=3)
            if facts:
                system+="\n\n关于用户记忆\n"+"\n".join(facts)
        messages=[{"role":"system","content":system}]
        messages.extend(self.history)
        messages.append({"role":"user","content":user_input})
        return messages


    def _extract_memories(self, user_input:str, answer:str):
        """对话结束后，让 LLM 提取值得长期记住的事实。"""
        if not self.memory:
            return
        prompt = f"""从以下对话中提取值得长期记住的关于用户的事实（偏好、身份、项目等）。
        如果没有值得记的，只输出"无"。
        每行一条，不要加序号或符号。

        用户: {user_input}
        助手: {answer}"""

        result=self.llm.chat([{"role":"user","content":prompt}]).content or ""
        if result.strip() =="无":
            return
        for line in result.strip().split("\n"):
            line = line.strip("- •*").strip()
            if line and line != "无":
                added = self.memory.add(line)
                if added:
                    print(f"  💾 记住：{line}")
                else:
                    print(f"  ⏭️ 跳过重复：{line}")


    def _maybe_compress(self):
        """历史太长时，把旧的部分摘要成一段话。"""
        total_chars = sum(len(str(h.get("content", ""))) for h in self.history)
        if total_chars < self.context_limit:
            return
        old = self.history[:-4]
        recent = self.history[-4:]
        old_text = "\n".join(
            f"{h['role']}: {h.get('content', '')}" for h in old
        )
        prompt = f"请用一段话总结以下对话，保留关键信息和结论：\n{old_text}"
        new_summary = self.llm.chat([{"role": "user", "content": prompt}]).content or ""
        self.summary = (self.summary + "\n" + new_summary).strip()
        self.history = recent
        print(f"  📦 上下文压缩：{len(old)} 条 → 摘要，保留最近 {len(recent)} 条")

















