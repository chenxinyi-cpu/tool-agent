"""用 DeepSeek API 批量生成医疗问诊 SFT 数据。
断点续跑：脚本会记录已生成的数量，中断后重跑会从断点继续。
"""
import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

# 从 .env 加载环境变量
load_dotenv()

# ---------- 配置 ----------
API_KEY = os.environ.get("DEEPSEEK_API_KEY")
if not API_KEY:
    raise SystemExit("请先在 .env 里配置 DEEPSEEK_API_KEY")

CLIENT = OpenAI(api_key=API_KEY, base_url="https://api.deepseek.com/v1")
MODEL = "deepseek-chat"

OUTPUT_FILE = Path(r"E:\tool_agent/medical_sft.json")
BATCH_SIZE = 10          # 每次生成 10 条
TOTAL_TARGET = 200       # 总共要 200 条
SLEEP_BETWEEN = 1.0      # 每次调用间隔（防止限流）

# ---------- Prompt 模板 ----------
PROMPT_TEMPLATE = """你是一个医疗问诊数据生成器。请生成 {n} 条真实用户会问的医疗问题，并给出专业回答。

要求：
1. 问题要像真实用户问的（口语化、有具体症状描述）
2. 覆盖不同科室：内科、外科、儿科、妇科、皮肤科、眼科、耳鼻喉科、神经科等
3. 每个问题不同，不要重复
4. 回答要专业严谨，用通俗语言，包含：可能的原因、建议的处理方式、何时该就医
5. 回答末尾必须包含这一句："以上建议仅供参考，具体请咨询医生。"
6. 输出必须是纯 JSON 数组，格式如下（不要加 markdown 代码块）：

[
  {{"instruction": "用户的问题", "input": "", "output": "专业回答"}},
  ...
]

请生成 {n} 条：
"""

# ---------- 主逻辑 ----------
def load_existing():
    """读已有的数据，用于断点续跑。"""
    if OUTPUT_FILE.exists():
        with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                print("⚠️ 已有文件损坏，从零开始")
                return []
    return []


def save(data):
    """保存数据（原子写入，防止中途崩溃损坏）。"""
    tmp = OUTPUT_FILE.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    tmp.replace(OUTPUT_FILE)


def generate_batch(n):
    """调用 DeepSeek 生成一批数据。"""
    resp = CLIENT.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "user", "content": PROMPT_TEMPLATE.format(n=n)}
        ],
        temperature=1.0,        # 高温度，增加多样性
        response_format={"type": "json_object"},   # 让 DeepSeek 返回 JSON
    )
    content = resp.choices[0].message.content

    # DeepSeek 返回 {"data": [...]} 或直接 [...]，都要兼容
    try:
        parsed = json.loads(content)
        if isinstance(parsed, dict):
            # 找 dict 里第一个 list
            for v in parsed.values():
                if isinstance(v, list):
                    return v
            return []
        return parsed
    except json.JSONDecodeError as e:
        print(f"  ⚠️ JSON 解析失败：{e}")
        return []


def dedup(existing, new_items):
    """按 instruction 去重。"""
    seen = {item["instruction"] for item in existing}
    result = []
    for item in new_items:
        # 校验字段
        if not all(k in item for k in ("instruction", "input", "output")):
            continue
        if not item["instruction"].strip() or not item["output"].strip():
            continue
        if item["instruction"] in seen:
            continue
        seen.add(item["instruction"])
        result.append(item)
    return result


def main():
    data = load_existing()
    print(f"已有 {len(data)} 条数据")

    while len(data) < TOTAL_TARGET:
        print(f"\n生成一批（当前 {len(data)}/{TOTAL_TARGET}）...")
        try:
            batch = generate_batch(BATCH_SIZE)
        except Exception as e:
            print(f"  ❌ API 调用失败：{e}")
            print("  等待 10 秒后重试...")
            time.sleep(10)
            continue

        new_items = dedup(data, batch)
        print(f"  拿到 {len(batch)} 条，去重后新增 {len(new_items)} 条")

        if not new_items:
            print("  本批全是重复，跳过")
            time.sleep(SLEEP_BETWEEN)
            continue

        data.extend(new_items)
        save(data)
        print(f"  已保存，当前总数：{len(data)}")

        time.sleep(SLEEP_BETWEEN)

    print(f"\n✅ 完成！共 {len(data)} 条数据，保存到 {OUTPUT_FILE}")


if __name__ == "__main__":
    main()