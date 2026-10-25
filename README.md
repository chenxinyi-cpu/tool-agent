# My Agent

一个从零实现的多步推理 Agent，**不依赖 LangChain/LangGraph 等上层 Agent 框架**。个人技术练习项目，同时配套医疗领域 QLoRA‑SFT 微调实验，完成从 Agent 框架搭建、MCP 工具调度、RAG 检索增强到垂直领域模型微调完整技术闭环。

##  特性

- **ReAct 循环**：基于 Function Calling 的工具调用，比正则解析稳定
- **多步推理**：能自动完成"查天气 → 搜景点 → 搜价格 → 算预算 → 输出行程"的完整链路
- **RAG 检索增强**：基于 Chroma 向量数据库实现文档切片、向量化、语义检索，为 Agent 注入外部知识库
- **上下文压缩**：对话历史超 8000 字符时自动摘要旧消息
- **长期记忆**：跨会话记住用户偏好，JSON 持久化
- **语义去重**：新记忆进库前，用 embedding 计算和已有记忆的相似度，> 0.85 视为重复（经过多组实验调优确定阈值）
- **向量检索**：记忆检索用 embedding 余弦相似度，支持"语义相近但字面不同"的匹配
- **Streamlit Web UI**：浏览器聊天界面
- **单元测试 & 自动化 Eval 评测流水线**：pytest 覆盖核心模块；搭建 12 条 Case 的 Agent 效果评测集
- **错误重试**：指数退避重试 LLM 调用
- **Skill 系统**：按需加载流程知识（如医疗问诊），参考 Claude Code 设计
- **MCP 协议集成**：3 个独立 MCP Server（Weather/Travel/Math），支持跨应用工具复用

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
          ↓
┌──────────────────────┐
│  MCP Adapter         │  同步/异步桥接
│  (mcp_adapter.py)    │
└──────────┬───────────┘
           ↓
   ┌───────┼───────┐
   ↓       ↓       ↓
┌──────┐┌──────┐┌──────┐
│Weather││Travel││ Math │  MCP Servers
│Server ││Server││Server│
└──────┘└──────┘└──────┘
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

**执行 Agent 自动化 Eval 评测流水线**

```bash
python eval_pipeline.py
```

**医疗 SFT 模型微调脚本**

```bash
cd SFT
python sft_train.py
```

##  使用示例

```
你: 帮我规划去成都玩 3 天，预算中等

[Agent 自动执行]
 get_weather({'city': '成都'})                    → WeatherServer (MCP)
 search_attractions({'city': '成都', 'days': 3}) → TravelServer  (MCP)
 estimate_budget({'city': '成都', 'days': 3, 'level': '中等'}) → TravelServer (MCP)
 calculator({'expression': '346*2 + 200*3 + 310'}) → MathServer  (MCP)

助手: 【成都 3 天行程规划】
Day 1: 熊猫基地 + 市区文化
Day 2: 武侯祠 + 锦里古街
Day 3: 宽窄巷子 + 三星堆
预算明细：约 ¥1,602/人
```

## 项目结构

### Part 1: Agent 框架（手写实现，不依赖上层 Agent 库）

- ReAct + Function Calling
- RAG 全链路（Chroma 向量库、文档切片、语义检索）
- 上下文压缩 + 长期记忆（向量检索 + 语义去重）
- Skill 系统（按需加载流程知识）
- Streamlit Web UI
- 自动化 Eval 评测流水线 `eval_pipeline.py`
- 详细的代码在根目录

### Part 2: SFT 微调（医疗领域，SFT / 目录）

基于 Qwen2.5‑0.5B 模型，采用 QLoRA 完成指令微调，面向患者科普问诊场景

- 数据集：通过大模型 API 批量生成 200 条医疗问答样本；自研脚本完成样本去重、格式校验；Prompt 强制输出医疗免责声明
- 训练配置：4‑bit 量化 + LoRA 微调；仅训练总参数量 0.88%；训练显存占用降至 2GB；单卡 RTX4090 完成 75 步训练耗时 47s
- 实验指标：训练 Loss 由 2.63 下降至 1.79；修复原模型误读症状问题；模型可结构化输出「危险信号 + 护理建议 + 就医指导」；通过人工抽样校验医疗风险识别能力

> 
> 注意：本医疗模型仅用于技术实验，**不可用于真实临床诊断**

### Part 3: Skill 系统（流程知识复用）

- `SKILL.md` 定义可复用的流程（使用场景、回答步骤、输出格式、验收标准）
- Agent 启动时只加载 skill 名字 + 描述（省 token）
- 任务匹配时通过 `load_skill(name)` 工具按需加载完整内容
- 详细的代码在 `skill_loader.py` 和 `skills/` 目录

### Part 4: MCP 协议集成（工具标准化）

- 3 个独立 MCP Server（Weather / Travel / Math），各自负责一类工具
- MCP Adapter 做同步 / 异步桥接，自动转换 Schema 格式
- 工具从 "硬编码在 Agent 里" 变成 "可独立部署的 Server"
- 详细的代码在 `mcp_adapter.py` 和 `mcp_servers/` 目录

## 测试

```bash
pytest -v
```

预期：7 个单元测试全部通过。

```bash
# Agent自动化评测流水线
python eval_pipeline.py
```

- 评测集包含 **12 个 Case**，整体系统运行零报错；工具调用匹配率达到 **75%**
- 失败案例主要集中于**工具歧义选择场景**；完成「定位‑根因‑兜底」工程闭环分析
- 后续优化方向：增强工具描述、Few‑shot 提示优化，提升歧义场景下工具调用准确率

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

### 6. RAG 向量检索

基于 Chroma 向量数据库实现完整 RAG 链路：文档切片、文本向量化、语义检索，为 Agent 提供外部知识库注入能力；记忆检索从 "字符交集" 升级到 "embedding 余弦相似度"：

- query 和所有记忆都转成向量
- 按余弦相似度排序，返回 top‑k
- 有 "语义近似" 的匹配能力（"跑步" 能匹配 "运动"）

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

### 8. MCP 集成（同步/异步桥接）

**核心挑战**：Agent 是同步的（普通函数调用），MCP 是异步的（`async/await`）。需要一个桥接层。

**解决方案**：
- 用**后台 asyncio 线程**维持所有 MCP 连接
- 主线程通过 `asyncio.run_coroutine_threadsafe` 调用后台异步 API
- 用 `AsyncExitStack` 管理所有连接的生命周期

**Schema 自动转换**：
- MCP 格式：`{name, description, inputSchema}`
- OpenAI 格式：`{type: "function", function: {name, description, parameters}}`
- 适配器自动完成字段重命名 + 嵌套包装 + 空值兜底

**路由机制**：
- 启动时建立 `tool_name → server_name` 路由表
- LLM 决定调用后，Agent 查表找到对应的 MCP Server
- 支持多 Server 协作（一个 Agent 可同时挂载 3 个 Server）

**为什么用 MCP**：
- **可复用**：Weather Server 可以被任何支持 MCP 的应用使用（Claude Desktop、Cursor、我的 Agent）
- **可解耦**：每个 Server 独立部署、独立升级
- **标准化**：解决 M×N 集成问题，降到 M+N

## 后续迭代规划

1. 结合本人硕士研究方向：将 ABUS 三维乳腺超声图像分割模型封装为 MCP 工具，拓展医疗 Agent 图像分析能力
2. 扩充 Eval 评测 Case 数量，新增任务完成率、幻觉检测等评测指标
3. 优化歧义场景下工具选择逻辑，进一步提升工具调用匹配率
4. 优化 RAG 切片策略，提升长文档知识库问答效果

##  参考

实现过程中对比学习了三个开源 Agent 框架：

- [pi](https://github.com/earendil-works/pi)：可扩展的 hook 系统
- [nanobot](https://github.com/HKUDS/nanobot)：Dream 记忆系统
- [dsh](https://github.com/deepseek-ai/deepseek-harness)：插件化架构

## License

MIT
 
> 声明：本仓库全部为个人学习练习项目；其中医疗相关模型、脚本仅用于技术实验，严禁直接用于临床诊断。