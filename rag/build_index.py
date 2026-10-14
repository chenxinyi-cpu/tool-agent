"""构建 RAG 知识库：解析 medical_sft.json + 项目文档，存入 Chroma。

用法：
    cd tool_agent
    python rag/build_index.py
"""
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

from llm import LLMClient


# ============ 两套分隔符，按文档类型分开 ============
# 医疗问答：没有 Markdown 标题，用中文标点
MEDICAL_SEPARATORS = ["\n\n", "\n", "；", "。", "，", " "]
# 项目文档：有 Markdown 标题，优先按标题切
MARKDOWN_SEPARATORS = ["\n## ", "\n### ", "\n\n", "\n", "。", " "]


class DashScopeEmbeddings(Embeddings):
    """把项目的 LLMClient.embed 包装成 LangChain Embeddings 接口。

    注意：text-embedding-v3 支持批量输入（单次最多 25 条），
    批量调用比逐条调用快得多。
    """

    def __init__(self, llm: LLMClient, batch_size: int = 10):
        self.llm = llm
        self.client = llm.embed_client  # 复用已配置的 client
        self.batch_size = batch_size

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        results = []
        total = len(texts)
        for i in range(0, total, self.batch_size):
            batch = texts[i:i + self.batch_size]
            resp = self.client.embeddings.create(
                model="text-embedding-v3",
                input=batch,
            )
            results.extend([d.embedding for d in resp.data])
            done = min(i + self.batch_size, total)
            print(f"  向量化进度：{done}/{total}")
        return results

    def embed_query(self, text: str) -> list[float]:
        resp = self.client.embeddings.create(
            model="text-embedding-v3",
            input=text,
        )
        return resp.data[0].embedding


def load_medical_sft(path: Path, max_chunk: int = 800) -> list[Document]:
    """把 medical_sft.json 的每条问答对转成 Document。

    策略：一问一答作为一个完整 chunk，不切碎（保护上下文完整性）。
    只有 output 超过 max_chunk 字时，才按段落二次切分，
    并且每个分片都带上原问题作为上下文。
    """
    if not path.exists():
        print(f"⚠️ 找不到 {path}，跳过医疗数据")
        return []

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 只在长文本时才用到的 splitter（用医疗专用分隔符）
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=max_chunk,
        chunk_overlap=100,
        separators=MEDICAL_SEPARATORS,
    )

    docs = []
    for i, item in enumerate(data):
        instruction = (item.get("instruction") or "").strip()
        output = (item.get("output") or "").strip()
        if not instruction or not output:
            continue

        if len(output) <= max_chunk:
            # 短回答：整体作为一个 Document
            docs.append(Document(
                page_content=f"问题：{instruction}\n\n回答：{output}",
                metadata={
                    "source": "medical_sft",
                    "doc_id": i,
                    "question": instruction,
                },
            ))
        else:
            # 长回答：按段落切，每个 chunk 都带原问题
            for j, chunk in enumerate(splitter.split_text(output)):
                docs.append(Document(
                    page_content=f"问题：{instruction}\n\n回答片段：{chunk}",
                    metadata={
                        "source": "medical_sft",
                        "doc_id": i,
                        "chunk_id": j,
                        "question": instruction,
                    },
                ))
    return docs


def load_markdown_docs(paths: list[Path]) -> list[Document]:
    """加载项目文档（README、SKILL.md 等），按 Markdown 标题切分。"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=80,
        separators=MARKDOWN_SEPARATORS,
    )

    docs = []
    for p in paths:
        if not p.exists():
            print(f"⚠️ 找不到 {p}，跳过")
            continue
        text = p.read_text(encoding="utf-8")
        for i, chunk in enumerate(splitter.split_text(text)):
            docs.append(Document(
                page_content=chunk,
                metadata={"source": p.name, "chunk_id": i},
            ))
    return docs


def build():
    print("=" * 60)
    print("构建 RAG 知识库")
    print("=" * 60)

    project_root = Path(__file__).parent.parent
    llm = LLMClient(model="deepseek-chat")

    # ---------- 1. 医疗问答 ----------
    # 兼容两种路径：SFT/medical_sft.json 或 根目录/medical_sft.json
    medical_candidates = [
        project_root / "SFT" / "medical_sft.json",
        project_root / "medical_sft.json",
    ]
    medical_path = next((p for p in medical_candidates if p.exists()), None)
    if medical_path:
        medical_docs = load_medical_sft(medical_path)
        print(f"✓ 医疗问答：{len(medical_docs)} 个文档（来源：{medical_path.name}）")
    else:
        medical_docs = []
        print("⚠️ 未找到 medical_sft.json")

    # ---------- 2. 项目文档 ----------
    md_paths = [
        project_root / "README.md",
        project_root / "skills" / "medical" / "SKILL.md",
    ]
    md_docs = load_markdown_docs(md_paths)
    print(f"✓ 项目文档：{len(md_docs)} 个文档")

    all_docs = medical_docs + md_docs
    if not all_docs:
        print("❌ 没有可索引的文档，退出")
        return

    print(f"\n总计 {len(all_docs)} 个文档，开始向量化...")

    # ---------- 3. 存 Chroma ----------
    persist_dir = project_root / "chroma_db"
    if persist_dir.exists():
        shutil.rmtree(persist_dir)
        print(f"  已清空旧向量库：{persist_dir}")

    embeddings = DashScopeEmbeddings(llm)

    vectorstore = Chroma.from_documents(
        documents=all_docs,
        embedding=embeddings,
        persist_directory=str(persist_dir),
        collection_name="agent_kb",
    )

    print(f"\n✓ 向量库已保存到 {persist_dir}")
    print(f"  集合大小：{vectorstore._collection.count()}")


if __name__ == "__main__":
    build()