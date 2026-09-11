"""Mini-GPT - AI Coding Assistant by Anuj Mhatre
Lightweight version for Render free tier.
"""

import os
import gradio as gr

MODEL_ID = "Qwen/Qwen2.5-Coder-1.5B-Instruct"

SYSTEM_PROMPT = """You are Mini-GPT, a friendly AI coding assistant created by Anuj Mhatre.
You specialize in programming, coding, and software development.
You are knowledgeable, concise, and always ready to help with code."""

model = None
tokenizer = None

def load_model():
    global model, tokenizer
    if model is None:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        print("Loading Mini-GPT model...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            device_map="cpu",
            trust_remote_code=True,
        )
        print("Model loaded!")
    return model, tokenizer

def chat(message, history):
    model, tokenizer = load_model()
    
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for h in history:
        messages.append({"role": "user", "content": h[0]})
        if h[1]:
            messages.append({"role": "assistant", "content": h[1]})
    messages.append({"role": "user", "content": message})
    
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    
    outputs = model.generate(
        **inputs,
        max_new_tokens=256,
        temperature=0.7,
        do_sample=True,
    )
    
    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return response

demo = gr.ChatInterface(
    fn=chat,
    title="Mini-GPT",
    description="AI Coding Assistant by Anuj Mhatre | Loading model on first message...",
    examples=["What are you?", "Write a Python hello world", "Explain async/await"],
    theme=gr.themes.Soft(),
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
