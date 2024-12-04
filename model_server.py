from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging
import time
from peft import PeftModel

logging.basicConfig(level=logging.INFO)

app = FastAPI()

class InputText(BaseModel):
    input_text: str

# model_name = "taronklm/Qwen2.5-0.5B-Instruct-lora-chatbot"
# model_name = "taronklm/Qwen2.5-0.5B-Instruct-chatbot"
model_name = "taronklm/trained_model"
# model_name = r"C:\Users\Taro\Desktop\bachelorarbeit\code\bot\fine_tuned_model_v1"
# model_name = r"C:\Users\Taro\Desktop\bachelorarbeit\code\bot\chatbot-backend\fine_tuned_model"
# model_name = "KingNish/Qwen2.5-0.5b-RBase"
# model_name = "HuggingFaceTB/SmolLM2-360M-Instruct"
# model_name = "Qwen/Qwen2.5-0.5B-Instruct-GGUF"
# model_name = "taronklm/Prompto"
# model_name = "Qwen/Qwen2.5-0.5B-Instruct"


model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)
print("Model vocab size:", model.config.vocab_size)

lora_model = PeftModel.from_pretrained(model, r"C:\Users\Taro\Desktop\bachelorarbeit\code\bot\fine_tuned_model_v1")

tokenizer = AutoTokenizer.from_pretrained(r"C:\Users\Taro\Desktop\bachelorarbeit\code\bot\fine_tuned_model_v1", trust_remote_code=True)
print("Tokenizer vocab size:", len(tokenizer))

model.resize_token_embeddings(len(tokenizer))  # Adjust embedding size
# model = PeftModel.from_pretrained(model, model_id=lora_path)

lora_model.eval()

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

# SYS_PROMPT = """You are an assistant specialized in creating and optimizing prompts."""

async def generate_prompt_response(prompt):
    start_time = time.time()

    messages = [{"role": "system", "content":SYS_PROMPT},{"role": "user","content": prompt}]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(lora_model.device)

    generated_ids = lora_model.generate(
        **model_inputs,
        max_new_tokens=128,
        temperature=0.1,
        do_sample=True,
        top_k=50,         
        top_p=0.9    
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    end_time = time.time()
    model_response_time = end_time - start_time

    print(f"Model respinse time: {model_response_time:.2f} seconds")

    print("RESPONSE: ",response)
    return response

@app.post("/generate")
async def generate_endpoint(input_data: InputText):
    input_text = input_data.input_text

    print("INPUT TEXT: ", input_text)
    
    if not input_text:
        raise HTTPException(status_code=400, detail="Input text is required")
    
    generated_text = await generate_prompt_response(input_text)

    print("GENERATED TEXT: ", generated_text)
    return {"generated_text": generated_text}

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000)