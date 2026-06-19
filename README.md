<div align="center">

# 🔧 ToolForge

### Fine-Tuning Small LLMs for Reliable Tool Calling

[![HuggingFace Model](https://img.shields.io/badge/🤗%20Model-toolforge--qwen2.5--1.5b--lora-yellow)](https://huggingface.co/Ahmed-Samir-Abdel-fattah/toolforge-qwen2.5-1.5b-lora)
[![HuggingFace Spaces](https://img.shields.io/badge/🚀%20Demo-Live%20on%20Spaces-blue)](https://huggingface.co/spaces/Ahmed-Samir-Abdel-fattah/toolforge)
[![GitHub](https://img.shields.io/badge/GitHub-ToolForge-black?logo=github)](https://github.com/ahmedsamir42003/ToolForge-LLM-Fine-Tuning-for-Tool-Calling)
[![License](https://img.shields.io/badge/License-Apache%202.0-green)](https://opensource.org/licenses/Apache-2.0)

*Fine-tuned Qwen2.5-1.5B-Instruct with QLoRA on 2,500 tool-calling examples — achieving 99% format accuracy and a 7-point gain in function-name accuracy over the base model.*

</div>

---

## 🎯 What is ToolForge?

Tool calling (also called function calling) is the ability of an LLM to decide **when** to call an external function and **how** to structure the call as valid JSON. It's a critical capability for building AI agents, copilots, and automated pipelines — but small open-source models often struggle to do it reliably out of the box.

**ToolForge** fine-tunes **Qwen2.5-1.5B-Instruct** using **QLoRA** so it can:

- 📦 Emit perfectly structured `<tool_call>` JSON blocks
- 🎯 Select the right function from a list of available tools
- 🔀 Handle parallel calls (multiple tools in one response)
- 🚫 Know when **not** to call a tool and answer directly in plain text
- 🛡️ Avoid hallucinating tools that weren't provided

---

## 📊 Results

Evaluated on a held-out test set of **200 examples** from [`NousResearch/hermes-function-calling-v1`](https://huggingface.co/datasets/NousResearch/hermes-function-calling-v1).

| Metric | Base Qwen2.5-1.5B | ToolForge (QLoRA) | Δ |
|---|:---:|:---:|:---:|
| 📐 Format Accuracy | 90.0% | **99.0%** | +9.0 pp |
| 🎯 Function Accuracy | 65.5% | **72.5%** | +7.0 pp |
| ⚙️ Argument F1 | 0.689 | **0.734** | +0.045 |
| 🛡️ Hallucination Rate | 0.0% | 0.5% | +0.5% |

> Hallucination rate: lower is better. All other metrics: higher is better.

---

## 🔍 Example Output

**User query:**
```
What's the weather in Cairo and convert 100 USD to EUR?
```

**Base model** — hallucinates results instead of calling tools:
```
The weather in Cairo is hot with high humidity. 100 USD = 75.43 EUR.
```

**ToolForge** — correctly emits structured tool calls:
```xml
<tool_call>
{"name": "get_weather", "arguments": {"city": "Cairo"}}
</tool_call>
<tool_call>
{"name": "convert_currency", "arguments": {"amount": 100, "from_currency": "USD", "to_currency": "EUR"}}
</tool_call>
```

---

## 🏗️ Architecture

```
NousResearch Dataset (11k examples)
         │
         ▼
  Sample 2,500 examples
  (train / test split)
         │
         ▼
  Qwen2.5-1.5B-Instruct
  + QLoRA (4-bit, r=16)
  + SFTTrainer (3 epochs)
         │
         ▼
  LoRA Adapter (~37MB)
  pushed to HuggingFace Hub
         │
         ▼
  Evaluation Pipeline
  (4 metrics, 200 test samples)
         │
         ▼
  Gradio Demo on HF Spaces
```

---

## ⚙️ Training Details

| Setting | Value |
|---|---|
| Base model | `Qwen/Qwen2.5-1.5B-Instruct` |
| Method | QLoRA (4-bit NF4 quantization) |
| LoRA rank / alpha | r=16, α=32 |
| Target modules | q_proj, k_proj, v_proj, o_proj |
| Training samples | 2,500 |
| Epochs | 3 |
| Batch size (effective) | 16 (4 × grad accum 4) |
| Learning rate | 2e-4 (cosine schedule) |
| Max sequence length | 2,048 tokens |
| Hardware | Kaggle T4 GPU (~2–4h) |
| Framework | HuggingFace PEFT + TRL SFTTrainer |

---

## 📐 Evaluation Metrics

The evaluation pipeline scores each prediction on four levels:

| Metric | What it measures |
|---|---|
| **Format Accuracy** | Does the output contain at least one parseable `<tool_call>` JSON block? |
| **Function Accuracy** | Does the model call the correct function(s) by name? |
| **Argument F1** | Token-level F1 on flattened key=value argument pairs |
| **Hallucination Rate** | Does the model call a function that wasn't in the provided tool list? |

---

## 🚀 Quick Start

### Load the adapter

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import torch

BASE_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
ADAPTER_REPO  = "Ahmed-Samir-Abdel-fattah/toolforge-qwen2.5-1.5b-lora"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
base = AutoModelForCausalLM.from_pretrained(BASE_MODEL_ID, quantization_config=bnb_config, device_map="auto")
model = PeftModel.from_pretrained(base, ADAPTER_REPO)
model.eval()
```

### Run inference

```python
import json

tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get current weather for a city",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"]
        }
    }
}]

system = (
    "You are a function calling AI model. You are provided with function signatures "
    "within <tools> </tools> XML tags.\n"
    f"<tools>\n{json.dumps(tools)}\n</tools>\n\n"
    "If the query requires a tool, output ONLY <tool_call> tags with no explanation. "
    "If it is a general question, reply in plain text."
)

messages = [
    {"role": "system", "content": system},
    {"role": "user",   "content": "What is the weather in Cairo?"}
]

prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    output = model.generate(**inputs, max_new_tokens=256, do_sample=False,
                            pad_token_id=tokenizer.pad_token_id)

new_tokens = output[0][inputs["input_ids"].shape[-1]:]
print(tokenizer.decode(new_tokens, skip_special_tokens=True))
# → <tool_call>{"name": "get_weather", "arguments": {"city": "Cairo"}}</tool_call>
```

---

## 📁 Repository Structure

```
ToolForge/
├── 📓 notebooks/
│   ├── 01_fine_tuning.ipynb       # QLoRA training on Kaggle
│   └── 02_evaluation.ipynb        # Base vs fine-tuned comparison
├── 🗂️ results/
│   ├── base_predictions.csv       # Row-level scores — base model
│   ├── finetuned_predictions.csv  # Row-level scores — ToolForge
├── 🤗 src/
|    ├──  🚀 app.py                 # Gradio demo
└── 📋 requirements.txt
```

---

## 🔗 Links

| Resource | Link |
|---|---|
| 🤗 Model on HuggingFace Hub | [Ahmed-Samir-Abdel-fattah/toolforge-qwen2.5-1.5b-lora](https://huggingface.co/Ahmed-Samir-Abdel-fattah/toolforge-qwen2.5-1.5b-lora) |
| 🚀 Live Demo | [huggingface.co/spaces/Ahmed-Samir-Abdel-fattah/toolforge](https://huggingface.co/spaces/Ahmed-Samir-Abdel-fattah/toolforge) |
| 📦 Training Dataset | [NousResearch/hermes-function-calling-v1](https://huggingface.co/datasets/NousResearch/hermes-function-calling-v1) |
| 🧠 Base Model | [Qwen/Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) |

---

<div align="center">

Built with ❤️ using 🤗 Transformers · PEFT · TRL · Gradio

</div>
