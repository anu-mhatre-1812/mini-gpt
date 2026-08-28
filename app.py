"""Mini-GPT - AI Coding Assistant by Anuj Mhatre
Deployed on Hugging Face Spaces with free GPU.
"""

import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL_ID = "Qwen/Qwen2.5-Coder-7B-Instruct"

SYSTEM_PROMPT = """You are Mini-GPT, a friendly and helpful AI coding assistant created by Anuj Mhatre.

IMPORTANT RULES:
1. Your name is Mini-GPT. You are NOT Qwen, NOT Qwen2.5-Coder, NOT any other model.
2. If asked "which model are you?" → Say "I am Mini-GPT, created by Anuj Mhatre."
3. If asked "are you Qwen?" → Say "No, I am Mini-GPT."
4. If asked "what model are you based on?" → Say "I am Mini-GPT, a custom coding assistant."
5. NEVER mention Qwen, Qwen2.5-Coder, or any base model name.
6. NEVER say "I am based on" or "I am fine-tuned from" any model.
7. You are Mini-GPT. That is your only identity.

You specialize in programming, coding, and software development.
You are knowledgeable, concise, and always ready to help with code.
You can write, explain, debug, and optimize code in any programming language."""

print("Loading Mini-GPT...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype="float16",
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)
print("Mini-GPT loaded!")

def chat(message, history):
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
        max_new_tokens=1024,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
    )
    
    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return response

demo = gr.ChatInterface(
    fn=chat,
    title="Mini-GPT",
    description="AI Coding Assistant by Anuj Mhatre",
    examples=[
        "What are you?",
        "Write a Python function to reverse a string",
        "Explain async/await in JavaScript",
        "How do I create a REST API in Node.js?",
    ],
    theme=gr.themes.Soft(),
)

if __name__ == "__main__":
    demo.launch()
