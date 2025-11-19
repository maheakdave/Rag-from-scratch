# RAG (Retrieval Augmented and Generation) from scratch

This repository is an educational project to learn about LLMs, implemented from scratch in Pytorch.

The UI uses chainlit as the chat application and the backend server uses FastAPI and Chromadb as it's vector database.

Current LLM in use - GPT2 (124 M).

Future Works - 
 - Implement other opensource SOTA LLMs (Llama 3, DeepSeekv2, Qwen).
 - Integrate MOE.
 - Use Triton inference server as the backend.
 - Write custom Triton kernels for efficient LLM inference.
 - Improve upon the current RAG implementation by using other advanced techniques for scalability.
   

To run the system, simply buid the docker containers, using the command - `docker compose up --build `
