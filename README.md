# ToolForge-LLM-Fine-Tuning-for-Tool-Calling

Great project idea — this is exactly the kind of thing that stands out. Let me give you the full plan.

---

## The Project in One Line

Fine-tune a small LLM to perform **structured tool-calling** using LoRA, then evaluate it against the base model with a clean metrics pipeline.

---

## 1. Model Choice

**Qwen2.5-1.5B-Instruct** or **Qwen2.5-3B-Instruct**

Why:
- Small enough to fine-tune on a free Colab/Kaggle GPU in hours
- Already has good instruction-following as a base
- Does NOT natively do reliable tool-calling → perfect for your goal
- Hugely popular in the industry right now (better CV signal than older models)
- Apache 2.0 license → no restrictions

---

## 2. Dataset Strategy

### Use this dataset: `NousResearch/hermes-function-calling-v1`
- Open source on HuggingFace
- ~11k tool-calling examples already formatted
- Covers: single tool call, parallel calls, nested reasoning, no-call cases
- You only need **2,000–3,000 samples** for a strong fine-tune at this scale

### Training example format (what each sample looks like)
```json
{
  "system": "You are a helpful assistant with access to tools...",
  "user": "What is the weather in Cairo and convert 25°C to Fahrenheit?",
  "assistant": "<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Cairo\"}}\n</tool_call>\n<tool_call>\n{\"name\": \"convert_temperature\", \"arguments\": {\"value\": 25, \"from\": \"C\", \"to\": \"F\"}}\n</tool_call>"
}
```

### If you want to build your own (optional, more impressive)
Use Claude/GPT to generate 500–1000 synthetic examples for a specific domain (e.g. finance tools, weather tools, calculator). This shows data engineering skills — mention it explicitly on your CV.

---

## 3. Fine-tuning Method

**QLoRA** — the only realistic choice for your constraints.

| Setting | Value |
|---|---|
| Base model | Qwen2.5-1.5B-Instruct |
| Method | QLoRA (4-bit quantization + LoRA adapters) |
| LoRA rank | r=16, alpha=32 |
| Target modules | q_proj, v_proj, k_proj, o_proj |
| Batch size | 4 (gradient accumulation 4 → effective 16) |
| Learning rate | 2e-4 with cosine schedule |
| Epochs | 3 |
| Max sequence length | 2048 |
| Hardware | Single T4 (Colab/Kaggle free tier) |
| Training time | ~2–4 hours |

### Libraries
```bash
pip install transformers peft trl bitsandbytes datasets accelerate
```

TRL's `SFTTrainer` handles almost everything — you won't be writing a training loop from scratch.

---

## 4. Evaluation Strategy

This is what separates a good portfolio project from a great one. Evaluate on **3 levels**:

| Level | Metric | What it measures |
|---|---|---|
| **Format** | % valid JSON tool calls | Does it produce parseable output? |
| **Function** | Function name accuracy | Does it call the right tool? |
| **Arguments** | Argument exact match / F1 | Are the arguments correct? |

Compare **base model vs fine-tuned model** on a held-out test set of 200 examples. The delta is your headline result.

Also add one qualitative metric: **hallucination rate** — how often does it call a tool that wasn't in the provided tool list?

---

## 5. Step-by-Step Timeline

### Day 1–2 — Setup & Data
```
- Set up Colab/Kaggle environment
- Load and explore NousResearch dataset
- Filter to 2500 train / 200 test samples
- Write preprocessing script (format into chat template)
- Verify tokenization looks correct
- Run base model inference on 20 examples → establish baseline
```

### Day 3–5 — Fine-tuning & Evaluation
```
- Configure QLoRA with PEFT
- Run SFTTrainer for 3 epochs
- Save LoRA adapter weights to HuggingFace Hub
- Write evaluation script (format accuracy, function accuracy, arg F1)
- Run eval on base model vs fine-tuned model
- Log results to a simple table / MLflow
```

### Day 6–7 — Demo & Polish
```
- Build a simple Gradio demo (user types a question + tool list → model outputs tool call)
- Write a clean README with results table, example outputs, architecture diagram
- Push everything to GitHub
- Write the CV bullet points
```

---

## 6. Demo (Day 6)

A Gradio interface takes ~50 lines:

```python
import gradio as gr

def predict(user_message, tools_json):
    # format prompt, run model, return structured output
    ...

gr.Interface(
    fn=predict,
    inputs=[
        gr.Textbox(label="User Message"),
        gr.Textbox(label="Available Tools (JSON)", lines=5)
    ],
    outputs=gr.Textbox(label="Tool Call Output")
).launch()
```

Deploy free on HuggingFace Spaces — gives you a live link to put on your CV.

---

## 7. Trade-offs

| Simplify this | Keep this |
|---|---|
| Skip multi-turn conversation handling | Keep the evaluation pipeline — it's the most impressive part |
| Use existing dataset, don't build from scratch | Keep the base vs fine-tuned comparison with clear numbers |
| Skip Celery/async serving | Keep the Gradio demo — it's a live proof |
| Skip RLHF/DPO | Keep QLoRA — it's the right industry technique |

---

## 8. Project Name

**ToolForge** — Fine-Tuning Small LLMs for Reliable Tool Calling

---

## 9. CV Entry

**ToolForge — LLM Fine-Tuning for Tool Calling**
07/2026 – 07/2026
[GitHub](#) | [Demo](#)

Fine-tuned Qwen2.5-1.5B using QLoRA on 2,500 tool-calling examples, improving structured output format accuracy from 31% to 89% and function-name accuracy from 44% to 91% over the base model.

- **Fine-tuning:** QLoRA (4-bit quantization, LoRA r=16), SFTTrainer, HuggingFace PEFT/TRL, Qwen2.5-1.5B-Instruct
- **Evaluation:** Format accuracy, function-name accuracy, argument F1, hallucination rate — base vs fine-tuned comparison on 200-sample held-out test set
- **Deployment:** Gradio demo on HuggingFace Spaces, LoRA adapter published to HuggingFace Hub

---

The numbers in the CV entry are placeholders — replace with your actual results. But based on similar projects, going from a weak base model to a QLoRA fine-tune on this task typically shows **40–60 percentage point improvements** on format accuracy, which is a strong headline number.
