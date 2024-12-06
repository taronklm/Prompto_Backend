from fastapi import APIRouter, HTTPException
from app.models import InputPrompt
from app.utils import generate_prompt_response

router = APIRouter(prefix="/chatbot")

@router.post("/")
def generate_response(input: InputPrompt):
    input_prompt = input.input_text

    if not input_prompt:
        raise HTTPException(status_code=400, detail="Input text is required.")
    
    generated_response = generate_prompt_response(input_prompt)

    return {"generated_text": generated_response}