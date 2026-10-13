"""Skill 加载器：扫描 skills/ 目录，加载 SKILL.md。"""
import re
from pathlib import Path


SKILLS_DIR = Path(__file__).parent / "skills"


def _parse_frontmatter(text: str) -> dict:
    """解析 SKILL.md 顶部的 YAML frontmatter（--- 之间）。"""
    match = re.match(r"^---\s*\n(.*?)\n---\s*\n", text, re.DOTALL)
    if not match:
        return {}
    meta = {}
    for line in match.group(1).split("\n"):
        if ":" in line:
            key, value = line.split(":", 1)
            meta[key.strip()] = value.strip()
    return meta


def list_skills() -> list:
    """列出所有 skill 的名字和描述，不加载内容。"""
    skills = []
    if not SKILLS_DIR.exists():
        return skills
    for skill_dir in SKILLS_DIR.iterdir():
        if not skill_dir.is_dir():
            continue
        skill_file = skill_dir / "SKILL.md"
        if not skill_file.exists():
            continue
        text = skill_file.read_text(encoding="utf-8")
        meta = _parse_frontmatter(text)
        skills.append({
            "name": meta.get("name", skill_dir.name),
            "description": meta.get("description", ""),
            "path": str(skill_file),
        })
    return skills


def load_skill(name: str) -> str:
    """加载一个 skill 的完整内容（用于注入上下文）。"""
    skill_dir = SKILLS_DIR / name
    skill_file = skill_dir / "SKILL.md"
    if not skill_file.exists():
        return f"错误：找不到 skill '{name}'"
    return skill_file.read_text(encoding="utf-8")


def format_skill_list_for_prompt() -> str:
    """把 skill 列表格式化成 system prompt 可用的字符串。"""
    skills = list_skills()
    if not skills:
        return ""
    lines = ["## 可用 Skills\n"]
    lines.append("当任务匹配以下 skill 时，调用 `load_skill(name)` 加载完整流程：\n")
    for s in skills:
        lines.append(f"- `{s['name']}`: {s['description']}")
    return "\n".join(lines)


if __name__ == "__main__":
    print("可用 skill 列表：")
    for s in list_skills():
        print(f"  - {s['name']}: {s['description']}")
    print()
    print("=" * 50)
    print("medical skill 的完整内容：")
    print(load_skill("medical")[:500] + "...")