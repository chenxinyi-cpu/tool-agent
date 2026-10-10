import json
import os
from datetime import datetime

import numpy as np


class Memory:
    # 去重阈值：相似度 > 此值认为是重复记忆
    DEDUP_THRESHOLD = 0.85
    # 检索阈值：相似度 < 此值不返回
    RECALL_THRESHOLD = 0.5

    def __init__(self,path:str="memory.json",llm=None):
        self.path=path
        self.embed_llm=llm
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

    def add(self,content:str,threshold: float = None)->bool:
        content=content.strip()
        if not content:
            return False
        if self.embed_llm is None:
            raise RuntimeError("Memory 需要传入 llm 才能做语义去重")

        threshold = threshold if threshold is not None else self.DEDUP_THRESHOLD

        # 算新记忆的 embedding
        try:
            new_emb = self.embed_llm.embed(content)
        except Exception as e:
            print(f"embed 失败，回退到精确去重：{e}")
            new_emb = None

        # 精确去重（无论如何都做一层，兜底）
        if any(f["content"] == content for f in self.facts):
            return False

        # 语义去重
        if new_emb is not None:
            for f in self.facts:
                if "embedding" not in f:
                    continue
                sim = self._cosine(f["embedding"], new_emb)
                if sim > threshold:
                    return False

        # 加新的
        entry = {
            "time": datetime.now().isoformat(timespec="seconds"),
            "content": content,
        }
        if new_emb is not None:
            entry["embedding"] = new_emb
        self.facts.append(entry)
        self._save()
        return True


    def recall(self,query:str,k:int=3)->list:
        if not self.facts:
            return []
        if self.embed_llm is None:
            return self._recall_by_chars(query, k)

        # 有 embed_llm → 走语义检索
        try:
            q_emb = self.embed_llm.embed(query)
        except Exception as e:
            print(f"  ⚠️ embed 失败，回退到字符匹配：{e}")
            return self._recall_by_chars(query, k)

        # 算相似度并排序
        scored = []
        for f in self.facts:
            if "embedding" not in f:
                continue
            sim = self._cosine(f["embedding"], q_emb)
            scored.append((sim, f["content"]))

        scored.sort(reverse=True)
        return [content for sim, content in scored[:k] if sim > self.RECALL_THRESHOLD]


    def _recall_by_chars(self,query:str,k:int=3)->list:
        """兜底：字符交集打分"""
        query_tokens = set(query)
        def score(f):
            return len(query_tokens & set(f["content"]))
        ranked = sorted(self.facts, key=score, reverse=True)
        return [f["content"] for f in ranked[:k] if score(f) > 0]


    @staticmethod
    def _cosine(a, b) -> float:
        """余弦相似度"""
        a = np.array(a, dtype=np.float32)
        b = np.array(b, dtype=np.float32)
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        if denom == 0:
            return 0.0
        return float(np.dot(a, b) / denom)


    def _save(self):
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self.facts, f, ensure_ascii=False, indent=2)


