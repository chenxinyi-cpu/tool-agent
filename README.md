# My Agent

一个从零实现的多步推理 Agent，不依赖任何 Agent 框架。

##  特性

- **ReAct 循环**：基于 Function Calling 的工具调用，比正则解析稳定
- **多步推理**：能自动完成"查天气 → 搜景点 → 搜价格 → 算预算 → 输出行程"的完整链路
- **上下文压缩**：对话历史超 8000 字符时自动摘要旧消息
- **长期记忆**：跨会话记住用户偏好，JSON 持久化
- **语义去重**：新记忆进库前，用 embedding 计算和已有记忆的相似度，> 0.85 视为重复
- **向量检索**：记忆检索用 embedding 余弦相似度，支持"语义相近但字面不同"的匹配
- **Streamlit Web UI**：浏览器聊天界面
- **单元测试**：pytest 覆盖核心模块
- **错误重试**：指数退避重试 LLM 调用
- **Skill 系统**：按需加载流程知识（如医疗问诊），参考 Claude Code 设计

##  架构

```
┌─────────────┐
│  Web UI     │  Streamlit
│  (web_ui.py)│
└──────┬──────┘
       ↓
┌─────────────┐
│  Agent      │  ReAct 循环 + 上下文压缩 + 记忆
│  (agent.py) │
└──┬───┬───┬──┘
   ↓   ↓   ↓
┌────┐┌─────┐┌──────┐
│LLM ││工具  ││记忆  │
│llm ││tools││memory│
└────┘└─────┘└──────┘
```

##  快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置 API Key

在项目根目录创建 `.env` 文件：

```
DEEPSEEK_API_KEY=sk-你的key
TAVILY_API_KEY=tvly-你的key（可选，用于搜索工具）
```

- DeepSeek Key: https://platform.deepseek.com/
- Tavily Key: https://tavily.com/

### 3. 运行

**命令行模式**：

```bash
python main.py
```

**Web UI 模式**：

```bash
streamlit run web_ui.py
```

浏览器会自动打开 `http://localhost:8501`。

##  使用示例

```
你: 帮我规划去成都玩 3 天，预算中等

[Agent 自动执行]
🔧 get_weather({'city': '成都'})
🔧 search_attractions({'city': '成都', 'days': 3})
🔧 estimate_budget({'city': '成都', 'days': 3, 'level': '中等'})
🔧 calculator({'expression': '346*2 + 200*3 + 310'})

助手: 【成都 3 天行程规划】
Day 1: 熊猫基地 + 市区文化
Day 2: 武侯祠 + 锦里古街
Day 3: 宽窄巷子 + 三星堆
预算明细：约 ¥1,602/人
```

## 项目结构

### Part 1: Agent 框架（手写）
- ReAct + Function Calling
- 上下文压缩 + 长期记忆（向量检索 + 语义去重）
- Skill 系统（按需加载流程知识）
- Streamlit Web UI
- 详细的代码在根目录

### Part 2: SFT 微调（医疗领域）
- QLoRA 微调 Qwen2.5-0.5B
- 200 条医疗问答数据
- 详细的代码在 `SFT/` 目录

### Part 3: Skill 系统（流程知识复用）
- `SKILL.md` 定义可复用的流程（使用场景、回答步骤、输出格式、验收标准）
- Agent 启动时只加载 skill 名字+描述（省 token）
- 任务匹配时通过 `load_skill(name)` 工具按需加载完整内容
- 详细的代码在 `skill_loader.py` 和 `skills/` 目录

### 两者的关系
Agent 负责"通用能力 + 工具调用"，SFT 负责"领域专精"。
实际部署时，可以根据场景选择：
- 通用场景 → Agent + DeepSeek
- 领域场景 + 隐私要求 → Agent + 微调后的本地模型

## 测试

```bash
pytest -v
```

预期：7 个测试全部通过。

##  技术要点

### 1. Function Calling（而不是正则解析）

早期版本用 prompt + 正则解析工具调用，LLM 输出格式不稳定。改用 Function Calling 后：
- 用 JSON Schema 定义工具
- 模型直接返回结构化 `tool_calls`
- 代码量减少一半，稳定性大幅提升

### 2. 上下文压缩

对话历史超过 8000 字符时，把最早的消息用 LLM 摘要成一段话，保留最近 4 条原文。

### 3. 长期记忆

对话结束后让 LLM 提取值得记的事实，存到 JSON；下次对话前检索相关记忆注入 system prompt。

### 4. 指数退避重试

LLM 调用失败时，等待 1s → 2s → 4s 后重试，最多 3 次。

### 5. 语义去重（防止记忆膨胀）

早期版本用精确字符串去重，但 LLM 每次提取措辞不同（"用户叫小明" vs "用户的名字是小明"），导致重复堆积。

改用 embedding 语义去重后：
- 每条记忆算一个 1024 维向量
- 新记忆入库前，和已有记忆逐一算余弦相似度
- 相似度 > 0.85 → 视为重复，跳过
- 实现"先精确、再语义"的检查顺序，省 API 调用

### 6. 向量检索

记忆检索从"字符交集"升级到"embedding 余弦相似度"：
- query 和所有记忆都转成向量
- 按余弦相似度排序，返回 top-k
- 有"语义近似"的匹配能力（"跑步"能匹配"运动"）

### 7. Skill 系统

Skill 是**可复用的流程知识**，和 Tool（能力）、Prompt（指令）不同：
- **Tool**：可调用的函数（如 `get_weather`）
- **Prompt**：一次性指令
- **Skill**：可复用的流程规范（写在 `SKILL.md` 里）

**工作机制**：
1. Agent 启动时只加载 skill 的名字+描述（省 token）
2. LLM 判断当前任务匹配哪个 skill
3. 通过 `load_skill(name)` 工具按需加载完整内容
4. 按 skill 定义的流程办事

**示例**：`skills/medical/SKILL.md` 定义了医疗问诊流程：
- 判断紧急程度
- 给出可能原因
- 分"在家护理"和"何时就医"建议
- 末尾加免责声明

**与 SFT 的区别**：

| 维度 | Skill | SFT |
|---|---|---|
| 改什么 | 输入上下文 | 模型参数 |
| 需要训练 | 不需要 | 需要 |
| 改需求成本 | 改文档（分钟） | 重新训练（小时） |
| 适合场景 | 流程外化 | 行为内化 |

##  参考

实现过程中对比学习了三个开源 Agent 框架：

- [pi](https://github.com/earendil-works/pi)：可扩展的 hook 系统
- [nanobot](https://github.com/HKUDS/nanobot)：Dream 记忆系统
- [dsh](https://github.com/deepseek-ai/deepseek-harness)：插件化架构

##  License

MIT