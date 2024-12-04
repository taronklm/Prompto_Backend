from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, GPT2Tokenizer, GPT2LMHeadModel
import logging
from peft import PeftModel

logging.basicConfig(level=logging.INFO)

app = FastAPI()

class InputText(BaseModel):
    input_text: str

# model_name = "distilgpt2"
# model_name = "taronklm/distilgpt2-lora-chatbot"
# model_name = "taronklm/distilgpt2-chatbot"
model_name = r"C:\Users\Taro\Desktop\bachelorarbeit\code\bot\distilgpt2-chatbot\checkpoint-69"

tokenizer = GPT2Tokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
print(f"Toenizer vocab size: {len(tokenizer)}")
model = GPT2LMHeadModel.from_pretrained(model_name)

model.resize_token_embeddings(len(tokenizer))

# print(f"Model vocab size: {len(model.transformer.wte.weight.size(0))}")


SYS_PROMPT = """You are an assistant specialized in creating and optimizing prompts.
You will be provided with either a subject and a context to generate a prompt, or an existing prompt to optimize.
Users will send you messages formatted as either:
1. "Optimize: ..." (for optimization)
2. "Subject: ... , Context: ..." (for prompt creation)
If the messages you recieved does not involve prompt creation or optimization, respond exactly with: 'I am only trained to create or optimize prompts. I cannot answer this.'
Do not provide any other responses or make up answers.
"""

# SYS_PROMPT = """You are an assistant specialized in creating and optimizing prompts."""

async def generate_prompt_response(prompt):
    instruction = f"{SYS_PROMPT} \n\n User: {prompt} \n Assistant:"

    inputs = tokenizer.encode(instruction, return_tensors="pt")
    outputs = model.generate(
        inputs=inputs,
        max_length=300,
        no_repeat_ngram_size=2,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=0.7,
        pad_token_id=tokenizer.eos_token_id
    )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Generated response: {response}")
    
    assistant_response = response.split("Assistant:")[-1].strip()
    print(f"Generated response: {assistant_response}")
    return assistant_response

@app.post("/generate")
async def generate_endpoint(input_data: InputText):
    try:
        input_text = input_data.input_text

        print("INPUT TEXT: ", input_text)
        
        if not input_text:
            raise HTTPException(status_code=400, detail="Input text is required")
        
        generated_text = await generate_prompt_response(input_text)

        print("GENERATED TEXT: ", generated_text)
        return {"generated_text": generated_text}
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000)