import os
import time
from openai import OpenAI

class LLMClient:
    def __init__(self,model:str="deepseek-chat"):
        self.model=model
        self.client=OpenAI(
            api_key=os.environ["DEEPSEEK_API_KEY"],
            base_url="https://api.deepseek.com/v1",
        )

    def chat(self,messages:list,tools:list=None,temperature:float=0.7,max_retries: int = 3):
        """
        调用 LLM，返回 assistant message 对象。
        - 如果模型决定调用工具，message.tool_calls 非空
        - 否则 message.content 是最终回答
        """
        kwargs={
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
        }
        if tools :
            kwargs["tools"]=tools

        last_exception = None
        for attempt in range(max_retries):
            try:
                response=self.client.chat.completions.create(**kwargs)
                return response.choices[0].message
            except Exception as e:
                last_exception = e
                if attempt == max_retries - 1:
                    raise
                wait = 2 ** attempt
                print(f"  ⚠️ LLM 调用失败（{type(e).__name__}），{wait} 秒后重试...")
                time.sleep(wait)
        raise last_exception

    def chat_stream(self,messages:list,temperature:float=0.7):
        """流式调用 LLM，逐块 yield 内容片段。不支持工具调用。"""
        stream=self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            stream=True,
        )
        for chunk in stream:
            if not chunk.choices:
                continue
            delta=chunk.choices[0].delta
            if delta.content:
                yield delta.content
