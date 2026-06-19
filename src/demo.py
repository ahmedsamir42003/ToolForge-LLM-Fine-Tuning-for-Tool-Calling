import gradio as gr
import json, torch, re
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

BASE_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
ADAPTER_REPO  = "Ahmed-Samir-Abdel-fattah/toolforge-qwen2.5-1.5b-lora"

# Load once at startup
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
base = AutoModelForCausalLM.from_pretrained(BASE_MODEL_ID, torch_dtype=torch.float16, device_map="auto")
model = PeftModel.from_pretrained(base, ADAPTER_REPO)
model.eval()

def predict(user_message, tools_json):
    try:
        tools = json.loads(tools_json)
    except Exception:
        return "❌ Invalid tools JSON"

    system = f"You are a function calling AI model. You are provided with function signatures within <tools> </tools> XML tags.\n<tools>\n{json.dumps(tools)}\n</tools>"
    messages = [{"role": "system", "content": system},
                {"role": "user",   "content": user_message}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=256, do_sample=False,
                                pad_token_id=tokenizer.pad_token_id)
    new_tokens = output[0][inputs["input_ids"].shape[-1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

EXAMPLE_TOOLS = json.dumps([{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get current weather for a city",
        "parameters": {"type": "object", "properties": {
            "city": {"type": "string", "description": "City name"}
        }, "required": ["city"]}
    }
}], indent=2)

gr.Interface(
    fn=predict,
    inputs=[
        gr.Textbox(label="User Message", placeholder="What is the weather in Cairo?"),
        gr.Textbox(label="Available Tools (JSON)", value=EXAMPLE_TOOLS, lines=12),
    ],
    outputs=gr.Textbox(label="Tool Call Output", lines=6),
    title="🔧 ToolForge — Structured Tool Calling",
    description="Fine-tuned Qwen2.5-1.5B-Instruct with QLoRA for reliable structured tool calling.",
    examples=[
        ["What is the weather in Cairo?", EXAMPLE_TOOLS],
        ["What's the temperature in Paris right now?", EXAMPLE_TOOLS],
    ]
).launch()
