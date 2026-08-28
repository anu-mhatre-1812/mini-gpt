"""Mini-GPT Inference - Uses system prompt for personality (no fine-tuning needed)."""

import sys
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen2.5-Coder-7B-Instruct"

# Mini-GPT Personality
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
You are knowledgeable, concise, and always ready to help with code."""

print("Loading Mini-GPT (Qwen2.5-Coder-7B)...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

# Load in float16 on CPU (no GPU needed)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype="auto",
    device_map="cpu",
    trust_remote_code=True,
)

print("Mini-GPT loaded!")

def chat(user_input):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_input},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    
    outputs = model.generate(
        **inputs,
        max_new_tokens=512,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
    )
    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return response

# Accept input from command line
if len(sys.argv) > 1:
    user_input = " ".join(sys.argv[1:])
    response = chat(user_input)
    print(f"\nMini-GPT: {response}")
else:
    print("\nUsage: python chat.py 'your question here'")
    print("Example: python chat.py 'What are you?'")
