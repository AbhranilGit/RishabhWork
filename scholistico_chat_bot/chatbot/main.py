from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from chat_api import ChatAPI

app = FastAPI()
chat_api = ChatAPI()

class Message(BaseModel):
    content: str

@app.post("/chat")
async def chat(message: Message):
    try:
        chat_api.add_message("user", message.content)
        response = chat_api.get_response()
        chat_api.add_message("assistant", response)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/set_model")
async def set_model(model: str):
    try:
        chat_api.set_model(model)
        return {"message": f"Model set to {model}"}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))