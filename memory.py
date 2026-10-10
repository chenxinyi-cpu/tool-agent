import json
import os
from datetime import datetime

class Memory:
    # 去重阈值：相似度 > 此值认为是重复记忆
    DEDUP_THRESHOLD = 0.85
    # 检索阈值：相似度 < 此值不返回
    RECALL_THRESHOLD = 0.5

    def __init__(self,path:str="memory.json",embed_llm=None):
        self.path=path
        self.embed_llm=embed_llm
        self.facts=[]
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    content = f.read().strip()
                    if content:  # ← 关键：空文件不解析
                        self.facts = json.loads(content)
            except Exception as e:  # ← 任何错误都不崩
                print(f"⚠️ 读取记忆失败，使用空记忆：{e}")
                self.facts = []

    def add(self,content:str,threshold: float = None):
        content=content.strip()
        if not content:
            return
        if any(f["content"] == content for f in self.facts):
            return
        self.facts.append({
            "time":datetime.now().isoformat(timespec="seconds"),
            "content":content,
        })
        with open(self.path,"w", encoding="utf-8") as f:
            json.dump(self.facts,f, ensure_ascii=False, indent=2)

    def recall(self,query:str,k:int=3)->list:
        if not self.facts:
            return []
        query_tokens = set(query)

        def score(fact):
            return len(query_tokens & set(fact["content"]))

        ranked = sorted(self.facts, key=score, reverse=True)
        # 只保留分数大于 0 的
        return [f["content"] for f in ranked[:k] if score(f) > 0]
