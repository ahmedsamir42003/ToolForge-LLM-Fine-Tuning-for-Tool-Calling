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

---

# ToolForge — Project Plan
**Fine-Tuning Small LLMs for Reliable Tool Calling**

---

## Overview

Fine-tune **Qwen2.5-1.5B-Instruct** using **QLoRA** on ~2,500 tool-calling examples, then evaluate it against the base model with a structured metrics pipeline and ship a live Gradio demo on HuggingFace Spaces.

**Platform:** Kaggle (free T4 GPU, up to 30h/week)
**Total estimated time:** 7 days

---

## Phase 1 — Environment & Data (Days 1–2)

**Goal:** Kaggle notebook ready, dataset loaded and preprocessed, baseline established.

### 1.1 Kaggle Setup
- Create a new Kaggle notebook, enable GPU (T4 x2 available)
- Install dependencies:
  ```bash
  pip install transformers peft trl bitsandbytes datasets accelerate huggingface_hub
  ```
- Log in to HuggingFace Hub from within the notebook:
  ```python
  from huggingface_hub import login
  login(token="YOUR_HF_TOKEN")  # store as Kaggle secret
  ```

### 1.2 Dataset Loading & Exploration
- Load `NousResearch/hermes-function-calling-v1` from HuggingFace
- Explore the dataset: inspect columns, count examples, review sample entries
- Understand the four sub-types: single tool call, parallel calls, nested reasoning, no-call cases

### 1.3 Data Preprocessing
- Filter and sample **2,500 train / 200 test** examples (stratified by sub-type)
- Convert each sample to Qwen2.5 chat template format:
  ```json
  {
    "system": "You are a helpful assistant with access to tools...",
    "user": "What is the weather in Cairo?",
    "assistant": "<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Cairo\"}}\n</tool_call>"
  }
  ```
- Tokenize and verify sequence lengths — cap at **2048 tokens**
- Save preprocessed splits to Kaggle dataset output for reuse

### 1.4 Baseline Evaluation
- Load the **base Qwen2.5-1.5B-Instruct** (no fine-tuning)
- Run inference on 20 test examples
- Manually score: does it produce valid JSON? Does it call the right tool?
- Record baseline numbers — these become your "before" in the results table

**Deliverable:** Preprocessed dataset saved + baseline numbers noted

---

## Phase 2 — Fine-Tuning (Days 3–4)

**Goal:** QLoRA fine-tuned model saved to HuggingFace Hub.

### 2.1 QLoRA Configuration
```python
# Key hyperparameters
bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4")

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
```

### 2.2 Training with SFTTrainer
```python
training_args = TrainingArguments(
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,   # effective batch size = 16
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    num_train_epochs=3,
    max_seq_length=2048,
    fp16=True,
    logging_steps=50,
    save_strategy="epoch",
    output_dir="./toolforge-output"
)
```
- Monitor training loss — expect it to drop significantly in epoch 1
- Watch for GPU OOM errors; reduce batch size if needed (use `gradient_accumulation_steps` to compensate)
- Estimated training time: **2–4 hours** on T4

### 2.3 Save Adapter Weights
- Save only the LoRA adapter (not the full model — much smaller):
  ```python
  model.save_pretrained("toolforge-lora-adapter")
  ```
- Push adapter to HuggingFace Hub:
  ```python
  model.push_to_hub("YOUR_USERNAME/toolforge-qwen2.5-1.5b-lora")
  ```

**Deliverable:** LoRA adapter live on HuggingFace Hub

---

## Phase 3 — Evaluation (Days 4–5)

**Goal:** Rigorous base vs. fine-tuned comparison with three metrics.

### 3.1 Evaluation Metrics

| Level | Metric | How to Compute |
|---|---|---|
| **Format** | % valid JSON tool calls | `json.loads()` — does it parse without error? |
| **Function** | Function name accuracy | Exact string match on `name` field |
| **Arguments** | Argument F1 | Token-level F1 on argument key-value pairs |
| **Hallucination** | Hallucination rate | % calls referencing a tool not in the provided list |

### 3.2 Evaluation Script Structure
```python
def evaluate_model(model, tokenizer, test_samples):
    results = []
    for sample in test_samples:
        prediction = generate(model, tokenizer, sample["prompt"])
        results.append({
            "format_valid": is_valid_json(prediction),
            "function_correct": check_function_name(prediction, sample["expected"]),
            "arg_f1": compute_arg_f1(prediction, sample["expected"]),
            "hallucinated": check_hallucination(prediction, sample["tools"])
        })
    return aggregate_metrics(results)
```

### 3.3 Run Evaluation on Both Models
- Run the same script on the **base model** and the **fine-tuned model**
- Record results in a comparison table (this is your headline result)
- Log with a simple CSV or MLflow if you prefer

### 3.4 Qualitative Analysis
- Pick 10 interesting examples: failures, edge cases, no-call cases
- Screenshot or write up the before/after — useful for the README and demo

**Deliverable:** Results table with base vs. fine-tuned numbers across all 4 metrics

---

## Phase 4 — Demo & Deployment (Days 6–7)

**Goal:** Live Gradio demo on HuggingFace Spaces + polished GitHub repo.

### 4.1 Gradio Demo
```python
import gradio as gr
from transformers import pipeline

def predict(user_message, tools_json):
    prompt = format_prompt(user_message, tools_json)
    output = pipe(prompt, max_new_tokens=256)[0]["generated_text"]
    return extract_tool_call(output)

gr.Interface(
    fn=predict,
    inputs=[
        gr.Textbox(label="User Message", placeholder="What is the weather in Cairo?"),
        gr.Textbox(label="Available Tools (JSON)", lines=6,
                   placeholder='[{"name": "get_weather", "parameters": {...}}]')
    ],
    outputs=gr.Textbox(label="Tool Call Output"),
    title="ToolForge — Structured Tool Calling Demo",
    examples=[
        ["What is the weather in Cairo?", '[{"name": "get_weather", ...}]'],
        ["Convert 25°C to Fahrenheit", '[{"name": "convert_temp", ...}]'],
    ]
).launch()
```

### 4.2 Deploy to HuggingFace Spaces
- Create a new Space (SDK: Gradio, hardware: CPU Basic — free)
- The demo loads the LoRA adapter from Hub at startup
- Gives you a live shareable URL: `huggingface.co/spaces/YOUR_USERNAME/toolforge`

### 4.3 GitHub Repository Structure
```
toolforge/
├── README.md                  # Project overview + results table
├── notebooks/
│   ├── 01_data_preprocessing.ipynb
│   ├── 02_fine_tuning.ipynb
│   └── 03_evaluation.ipynb
├── src/
│   ├── data_utils.py          # Dataset loading & formatting
│   ├── train.py               # QLoRA training script
│   ├── evaluate.py            # Metrics pipeline
│   └── demo.py                # Gradio app
├── results/
│   └── evaluation_results.csv
└── requirements.txt
```

### 4.4 README Must-Haves
- One-line description + badges (HF model, HF spaces, license)
- Results table: base model vs. fine-tuned across all 4 metrics
- Architecture diagram: data → QLoRA → adapter → eval → demo
- 2–3 example outputs (before/after comparison)
- How to reproduce (one-command setup)

**Deliverable:** Live demo link + clean GitHub repo ready to put on CV

---

## Summary Timeline

| Day | Phase | Key Output |
|---|---|---|
| 1 | Setup & Data Exploration | Kaggle env ready, dataset understood |
| 2 | Preprocessing & Baseline | Clean splits saved, baseline scores noted |
| 3 | Fine-tuning (run) | Training job executing on Kaggle GPU |
| 4 | Fine-tuning (done) + Eval setup | Adapter saved to Hub, eval script written |
| 5 | Evaluation | Results table: base vs. fine-tuned |
| 6 | Demo | Gradio app live on HF Spaces |
| 7 | Polish | README, GitHub, CV bullet ready |

---

## Kaggle-Specific Tips

- **Save aggressively** — Kaggle sessions time out. Save model checkpoints to `/kaggle/working/` and push to HF Hub after each epoch so you don't lose progress.
- **Use Kaggle Secrets** for your HF token (`Settings → Add-ons → Secrets`), never hardcode it.
- **Use T4 x2** if available — doubles VRAM to 30GB and speeds training significantly.
- **Dataset versioning** — save your preprocessed dataset as a Kaggle dataset output so you can load it in later notebooks without re-running preprocessing.
- **30h/week GPU limit** — don't waste quota on exploration. Do all data prep with CPU, only flip GPU on for actual training and inference.

---

## CV Bullet (fill in your real numbers)

**ToolForge — LLM Fine-Tuning for Tool Calling** | [GitHub](#) | [Demo](#)

Fine-tuned Qwen2.5-1.5B using QLoRA on 2,500 tool-calling examples, improving format accuracy from **XX% → XX%** and function-name accuracy from **XX% → XX%** over the base model.

- **Fine-tuning:** QLoRA (4-bit quantization, LoRA r=16), SFTTrainer, HuggingFace PEFT/TRL
- **Evaluation:** Format accuracy, function-name accuracy, argument F1, hallucination rate on 200-sample held-out test set
- **Deployment:** Live Gradio demo on HuggingFace Spaces, LoRA adapter published to HuggingFace Hub
