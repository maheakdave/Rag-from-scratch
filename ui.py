import httpx
import chainlit as cl
import os

FASTAPI_URL = os.getenv("FASTAPI_STREAM_URL", "http://127.0.0.1:8001/")

@cl.on_message
async def main(user_msg: str):
    msg = await cl.Message(content="").send()
    async with httpx.AsyncClient(timeout=None) as client:
        async with client.stream("GET", FASTAPI_URL, params={"text": user_msg}) as resp:
            async for chunk in resp.aiter_bytes():
                text = chunk.decode(errors="ignore")
                if text:
                    await msg.stream_token(text)
            await msg.update()
