from transformers import AutoTokenizer
import logging
import time
import psutil
from peft import AutoPeftModelForCausalLM

logging.basicConfig(level=logging.INFO)


adapter_model = "taronklm/trained_model"
base_model = "Qwen/Qwen2.5-0.5B-Instruct"

model = AutoPeftModelForCausalLM.from_pretrained(adapter_model)
# model = AutoModelForCausalLM.from_pretrained(base_model)
tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)

model.eval()
print("Model loaded")
SYS_PROMPT ="""
You are a prompt assistant. You must strictly adhere to the following rules:
1. For 'Optimize:' messages: Rephrase the prompt to make it clearer, more specific, and actionable, without changing its intent. 
    Prefix your response with 'Optimized Prompt:'.
2. For 'Subject:, Context:' messages: Create a new, effective prompt based on the details provided. 
    Prefix your response with 'Generated Prompt:'.
3. For all other inputs: Respond only with:
   'I am only trained to create or optimize prompts. I cannot answer this.'

Rules:
- Do not generate text beyond the scope of prompt creation or optimization.
- Do not provide explanations or additional context.

Failing to follow these guidelines is not allowed.
"""

def generate_prompt_response(prompt):
    process = psutil.Process()
    start_time = time.time()

    initial_memory = process.memory_info().rss / (1024 * 1024)
    initial_cpu = process.cpu_percent()

    initial_tokens = len(tokenizer.tokenize(prompt))
    print(tokenizer.tokenize(prompt))

    messages = [{"role": "system", "content":SYS_PROMPT},{"role": "user","content": prompt}]

    tokenized_chat = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        return_tensors="pt"
    )

    inputs = tokenizer(tokenized_chat, return_tensors="pt", add_special_tokens=False)

    outputs = model.generate(
        **inputs,
        max_new_tokens=128,
        temperature=0.1,
        do_sample=True,
        top_k=50,         
        top_p=0.9    
    )

    response = tokenizer.decode(outputs[0][inputs['input_ids'].size(1):], skip_special_tokens=True)

    generated_tokens = len(tokenizer.tokenize(response))

    print(tokenizer.tokenize(response))

    final_memory = process.memory_info().rss / (1024 * 1024)
    end_time = time.time()
    model_response_time = end_time - start_time
    final_cpu = process.cpu_percent()

    print(f"Model response time: {model_response_time:.2f} seconds")
    print(f"Final Memory Usage: {final_memory:.2f} MB")
    print(f"Final CPU Usage: {(final_cpu/6):.2f}%")
    print(f"Initial Token Count: {initial_tokens}")
    print(f"Generated Token Count: {generated_tokens}")

    return response