from GPT.gpt import get_llm,get_tokenizer,generate_with_greedy_typical
from GPT.config import Config
from fastapi import FastAPI
import torch
from fastapi.responses import StreamingResponse
import uvicorn

app = FastAPI()
model =  get_llm(category="gpt-2")
tokenizer = get_tokenizer("BPE")
GPT2_config = Config.GPT_CONFIG_124M

def stream_message(text):
    encoded_text = tokenizer.encode(text)
    encoded_text = torch.tensor(encoded_text).unsqueeze(0).cuda()
    for token_ids in generate_with_greedy_typical(model=model,tokens=encoded_text,max_new_tokens=100,context_size=GPT2_config["context_length"],stream_len=5):
        
        generated_text = tokenizer.decode(token_ids)
        yield generated_text

@app.get("/")
async def stream_endpoint(text:str):
    return StreamingResponse(stream_message(text=text),
                             media_type="text/plain; charset=utf-8")

if __name__ == '__main__':
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=False)