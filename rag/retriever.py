"""RAG 检索模块，对外提供 search_knowledge_base 工具函数。"""
import sys
from functools import lru_cache
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from langchain_chroma import Chroma

from llm import LLMClient
from rag.build_index import DashScopeEmbeddings


PERSIST_DIR = Path(__file__).parent.parent / "chroma_db"
COLLECTION_NAME = "agent_kb"


@lru_cache(maxsize=1)
def _get_vectorstore() -> Chroma:
    """懒加载向量库（只初始化一次，避免重复读盘）。"""
    llm = LLMClient(model="deepseek-chat")
    embeddings = DashScopeEmbeddings(llm)
    return Chroma(
        persist_directory=str(PERSIST_DIR),
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME,
    )


def search_knowledge_base(query: str, k: int = 3) -> str:
    """从知识库检索相关文档片段。

    适用场景：
    - 医疗健康类问题（症状、用药、就医建议）
    - 询问本项目的实现细节（MCP、Skill、记忆机制等）

    Args:
        query: 检索关键词或自然语言问题
        k: 返回的文档数量，默认 3

    Returns:
        格式化的检索结果字符串（含来源标注）
    """
    try:
        vs = _get_vectorstore()
    except Exception as e:
        return (
            f"知识库未初始化，请先运行 `python rag/build_index.py`。"
            f"（错误：{e}）"
        )

    try:
        docs = vs.similarity_search(query, k=k)
    except Exception as e:
        return f"检索失败：{e}"

    if not docs:
        return "知识库中没有找到相关内容"

    parts = []
    for i, d in enumerate(docs, 1):
        source = d.metadata.get("source", "未知")
        question = d.metadata.get("question", "")
        header = f"[{i}] 来源：{source}"
        if question:
            # 医疗问答：显示原问题，方便 LLM 判断相关性
            header += f" | 原问题：{question}"
        parts.append(f"{header}\n{d.page_content}")

    return "\n\n---\n\n".join(parts)


def search_knowledge_base_with_score(query: str, k: int = 3) -> str:
    """带相似度分数的版本，用于调试和评估。"""
    try:
        vs = _get_vectorstore()
        results = vs.similarity_search_with_score(query, k=k)
    except Exception as e:
        return f"检索失败：{e}"

    if not results:
        return "知识库中没有找到相关内容"

    parts = []
    for i, (d, score) in enumerate(results, 1):
        # Chroma 返回的是 L2 距离，越小越相关
        parts.append(
            f"[{i}] 距离={score:.3f} | 来源：{d.metadata.get('source', '未知')}\n"
            f"{d.page_content[:200]}..."
        )
    return "\n\n".join(parts)


if __name__ == "__main__":
    # 手动测试
    test_queries = [
        "半夜两三点醒来睡不着是什么原因",
        "MCP 适配器怎么解决同步异步问题",
        "孩子发烧要不要马上吃退烧药",
    ]
    for q in test_queries:
        print("=" * 60)
        print(f"Query: {q}")
        print("=" * 60)
        print(search_knowledge_base(q, k=2))
        print()