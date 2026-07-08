# RAG Project Environment Notes

## Project
document-intelligence-rag

## Python
Python 3.12 in WSL2 virtual environment

## Virtual environment
~/document-intelligence-rag/venv

## Confirmed working stack

- faiss-cpu: 1.14.3
- rank_bm25: installed and importable
- sentence-transformers: 5.6.0
- llama-index: importable
- ragas: 0.2.15
- FlagEmbedding: importable
- torch: 2.7.0+cu128
- CUDA available: True
- GPU: NVIDIA GeForce RTX 5060 Laptop GPU

## Important fix applied

The first PyTorch version pulled into the venv was torch 2.12.1+cu130, which failed because the system driver supported CUDA 12.8, not CUDA 13.0.

Fixed by reinstalling:

torch==2.7.0+cu128
torchvision==0.22.0
torchaudio==2.7.0

using the PyTorch CUDA 12.8 wheel index.

## Ragas note
ragas 0.3.9 and 0.4.3 failed due to a LangChain VertexAI import path that was
removed from langchain-community. Fixed by pinning langchain-community==0.2.19
+ langchain>=0.2.17,<0.3.0, plus a local shim module at
venv/lib/python3.12/site-packages/langchain_community/chat_models/vertexai.py
(not tracked in git — regenerate if the venv is rebuilt).

## Notes
This venv is for chunking, indexing, retrieval, evaluation, reranking, and
generation only. Extraction/checkpoint inference stays in the old
hybrid-doc-extractor venv — see PROJECT_GRAPH_REPORT.md for the two-venv split.

## Isolation decision

This RAG venv is separate from the old hybrid-doc-extractor venv.

The old extraction venv should remain untouched and should only be used for checkpoint inference or extraction-related scripts.

The RAG venv should be used for chunking, indexing, retrieval, evaluation, reranking, and generation.
