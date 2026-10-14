"""RAG 评测：用固定问题集测 Recall@k。

用法：
    cd tool_agent
    python rag/eval_rag.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from rag.retriever import _get_vectorstore


# 测试集：query + 期望命中的关键词（出现在正确 chunk 里）
TEST_CASES = [
    {
        "query": "半夜两三点醒来睡不着是什么原因",
        "expected_keywords": ["早醒", "失眠", "情绪"],
    },
    {
        "query": "孩子发烧38.5度要不要马上吃退烧药",
        "expected_keywords": ["退烧药", "对乙酰氨基酚", "38.5"],
    },
    {
        "query": "月经推迟10天验孕棒一条杠是怎么回事",
        "expected_keywords": ["月经", "验孕", "推迟"],
    },
    {
        "query": "眼睛干涩看电脑模糊怎么办",
        "expected_keywords": ["干眼", "人工泪液", "眨眼"],
    },
    {
        "query": "耳朵嗡嗡响是不是突发性耳聋",
        "expected_keywords": ["耳鸣", "听力", "耳鼻喉"],
    },
    {
        "query": "MCP 适配器怎么解决同步异步问题",
        "expected_keywords": ["MCP", "异步", "线程"],
    },
    {
        "query": "Skill 系统和 SFT 有什么区别",
        "expected_keywords": ["Skill", "SFT", "流程"],
    },
    {
        "query": "记忆的语义去重是怎么做的",
        "expected_keywords": ["去重", "embedding", "相似度"],
    },
    {
        "query": "胃疼反酸烧心是什么问题",
        "expected_keywords": ["反酸", "烧心", "胃"],
    },
    {
        "query": "脚踝扭伤应该冷敷还是热敷",
        "expected_keywords": ["冷敷", "48", "RICE"],
    },
]


def hit_at_k(docs, keywords) -> bool:
    """判断前 k 个文档里是否包含任一期望关键词。"""
    for d in docs:
        text = d.page_content
        if any(kw in text for kw in keywords):
            return True
    return False


def main():
    print("=" * 60)
    print("RAG 评测 - Recall@k")
    print("=" * 60)

    vs = _get_vectorstore()
    ks = [1, 3, 5]
    hits = {k: 0 for k in ks}
    failed = []

    for i, case in enumerate(TEST_CASES, 1):
        query = case["query"]
        keywords = case["expected_keywords"]

        # 取 top-5，然后分别算 Recall@1/3/5
        docs = vs.similarity_search(query, k=5)

        per_case_hit = {}
        for k in ks:
            hit = hit_at_k(docs[:k], keywords)
            per_case_hit[k] = hit
            if hit:
                hits[k] += 1

        status = "✅" if per_case_hit[3] else "❌"
        print(f"[{i}/{len(TEST_CASES)}] {status} {query}")
        if not per_case_hit[3]:
            top1 = docs[0].page_content[:80] if docs else "（无结果）"
            failed.append((query, keywords, top1))
            print(f"      期望关键词：{keywords}")
            print(f"      Top1：{top1}...")

    total = len(TEST_CASES)
    print("\n" + "=" * 60)
    print("评测结果")
    print("=" * 60)
    for k in ks:
        print(f"Recall@{k}: {hits[k]}/{total} ({hits[k]/total*100:.1f}%)")

    if failed:
        print(f"\n失败用例：{len(failed)} 个")
        for q, kw, top1 in failed:
            print(f"  - {q}")
            print(f"    期望：{kw}")
            print(f"    实际 Top1：{top1}")

    # 保存报告
    report = {
        "total": total,
        "recall": {f"@{k}": hits[k] / total for k in ks},
        "failed": [{"query": q, "keywords": kw} for q, kw, _ in failed],
    }
    out = Path(__file__).parent / "eval_rag_report.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n📄 报告已保存：{out}")


if __name__ == "__main__":
    main()