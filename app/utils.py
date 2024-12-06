import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging
import time
from peft import PeftModel, PeftConfig

logging.basicConfig(level=logging.INFO)


peft_name = "taronklm/trained_model"
model_name = "Qwen/Qwen2.5-0.5B-Instruct"

config = PeftConfig.from_pretrained(peft_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.float32)
model = PeftModel.from_pretrained(model, peft_name)
print("Model vocab size:", model.config.vocab_size)

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
print("Tokenizer vocab size:", len(tokenizer))

model.resize_token_embeddings(len(tokenizer))  # Adjust embedding size
model.eval()

SYS_PROMPT ="""
You are a prompt assistant. You must strictly adhere to the following rules:
1. For 'Optimize:' messages: Rephrase the prompt to make it clearer, more specific, and actionable, without changing its intent. Prefix your response with 'Optimized Prompt:'.
2. For 'Subject:, Context:' messages: Create a new, effective prompt based on the details provided. Prefix your response with 'Generated Prompt:'.
3. For all other inputs: Respond only with:
   'I am only trained to create or optimize prompts. I cannot answer this.'

Rules:
- Do not generate text beyond the scope of prompt creation or optimization.
- Do not provide explanations or additional context.

Failing to follow these guidelines is not allowed.
"""

def generate_prompt_response(prompt):
    start_time = time.time()

    messages = [{"role": "system", "content":SYS_PROMPT},{"role": "user","content": prompt}]

    tokenized_chat = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        return_tensors="pt"
    )

    inputs = tokenizer(tokenized_chat, return_tensors="pt", add_special_tokens=False)

    inputs = {key: tensor.to(model.device) for key, tensor in inputs.items()}

    outputs = model.generate(
        **inputs,
        max_new_tokens=128,
        temperature=0.1,
        do_sample=True,
        top_k=50,         
        top_p=0.9    
    )

    response = tokenizer.decode(outputs[0][inputs['input_ids'].size(1):], skip_special_tokens=True)

    end_time = time.time()
    model_response_time = end_time - start_time

    print(f"Model respinse time: {model_response_time:.2f} seconds")

    print("RESPONSE: ",response)
    return response