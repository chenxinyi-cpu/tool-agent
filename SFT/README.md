# 医疗问诊 SFT 微调

用 LoRA 微调 Qwen2.5-0.5B，做一个医疗问诊领域的对话模型。

## 为什么要做这个

基座模型（Qwen2.5-0.5B-Instruct）在通用对话上表现不错，但在医疗问诊这类专业场景：
- 回答不够结构化
- 缺少具体的危险信号提示
- 不会主动加免责声明

通过 LoRA 微调，用 200 条医疗问答数据，让模型学会医疗问诊的"专业风格"。

## 效果对比

**同一个问题**："我家宝宝2岁，昨晚发烧38.5度，还有点咳嗽，要不要马上去医院？"

**微调前**（基座模型）：
> 对于您家的宝宝的情况，以下是一些建议：...**虽然目前没有发烧或咳嗽的迹象**，但考虑到孩子的年龄（2岁），仍然建议及时就医...

❌ 模型读错了问题——用户明确说了"发烧+咳嗽"，模型却说"没有发烧的迹象"

**微调后**（LoRA 微调）：
> 孩子发热伴咳嗽要立即就医。建议先观察孩子的精神状态、呼吸是否急促或浅表、精神萎靡，如果体温超过38.5℃、嗜睡、抽搐、口唇青紫、呼吸困难...应尽快到医院儿科就诊。
>
> 在家护理方面：多喝水，注意休息，不要让孩子吃奶过多；可以物理降温...
>
> 以上建议仅供参考，具体请咨询医生。

✅ 结论先行 + 具体危险信号 + 护理建议 + 免责声明

完整对比见 `results/comparison.md`。

## 技术方案

| 项 | 值 |
|---|---|
| 基座模型 | Qwen2.5-0.5B-Instruct |
| 微调方法 | QLoRA（4-bit 量化 + LoRA） |
| LoRA rank | 8 |
| LoRA target | all（所有线性层） |
| 可训练参数 | 4.4M / 498M（0.88%） |
| 训练数据 | 200 条医疗问答（自造） |
| 训练轮数 | 3 |
| 学习率 | 1e-4（cosine 衰减） |
| 等效 batch size | 8（2 × 4 梯度累积） |

**为什么用 QLoRA**：
- 4-bit 量化让显存从 8GB 降到 2GB
- LoRA 只训练 0.88% 的参数，训练快，不易过拟合
- 适合小数据集（200 条）

## 数据构造

用 DeepSeek API 批量生成医疗问诊数据：
- 200 条问答对
- 覆盖内科、外科、儿科、妇科、皮肤科、眼科、耳鼻喉、神经科等
- 每条包含：可能原因、处理建议、就医时机、免责声明
- 脚本自动去重 + 格式校验，100% 合格

数据格式（Alpaca）：
```json
{
  "instruction": "用户问题",
  "input": "",
  "output": "专业回答"
}
```

## 项目结构

```
sft/
├── README.md                    # 本文件
├── train_medical.yaml           # 训练配置
├── generate_data.py             # 数据生成脚本
├── medical_sft.json             # 训练数据（200 条）
├── lora/                        # LoRA 权重
│   ├── adapter_config.json
│   └── adapter_model.safetensors   (~17MB)
└── results/
    ├── training_loss.png        # 训练 loss 曲线
    └── comparison.md            # 微调前后对比
```

## 复现步骤

**1. 环境准备**

```bash
pip install llamafactory
pip install openai  # 用于生成数据
```

**2. 生成训练数据**（可选，已提供 medical_sft.json）

```bash
export DEEPSEEK_API_KEY=sk-你的key
python generate_data.py
```

**3. 下载基座模型**

```bash
python -c "
from huggingface_hub import snapshot_download
snapshot_download('Qwen/Qwen2.5-0.5B-Instruct', local_dir='./Qwen2.5-0.5B-Instruct')
"
```

**4. 训练**

修改 `train_medical.yaml` 里的 `model_name_or_path` 指向你的模型目录，然后：

```bash
llamafactory-cli train train_medical.yaml
```

**5. 推理**

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

base = './Qwen2.5-0.5B-Instruct'
lora = './saves/qwen2.5-0.5b-medical'

tokenizer = AutoTokenizer.from_pretrained(base, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.bfloat16, device_map='cuda')
model = PeftModel.from_pretrained(model, lora)
model.eval()

messages = [{'role': 'user', 'content': '我最近老是半夜两三点醒，是不是肝有问题？'}]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(text, return_tensors='pt').to('cuda')

with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=300, temperature=0.7, do_sample=True)

response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
print(response)
```

## 技术要点

1. **QLoRA 微调**：4-bit 量化 + LoRA，显存友好
2. **小数据高效微调**：200 条数据，3 轮，不重复不遗忘
3. **数据构造**：用强模型（DeepSeek）生成弱模型（0.5B）的训练数据
4. **对话模板**：必须用 `qwen` 模板，否则模型学不会
5. **loss 掩码**：只对 assistant 的回答算 loss（system/user 部分的 label 设为 -100）

## 后续可做

- 扩大到 1000+ 条数据，loss 可降到 1.0 以下
- 尝试不同 `lora_rank`（4/16/32）对比效果
- 换成更接近真实数据的医疗对话数据（如 Huatuo-26M）
- 集成到 Agent 项目里，做"医疗问诊 Agent"