from pydantic import BaseModel


class InputPrompt(BaseModel):
    input_text: str
