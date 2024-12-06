from fastapi import APIRouter

from app.api.routes import chatbot

api_router = APIRouter()
api_router.include_router(chatbot.router)