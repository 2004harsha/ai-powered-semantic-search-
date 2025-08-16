# AI-powered PDF + Google Docs Semantic Search & Q&A

This is a Streamlit app that allows you to upload PDF or DOCX documents and ask semantic questions about their content using entirely free, open-source models running on CPU.

## Features

- Upload PDF or DOCX files
- Extract text from uploaded documents using `pdfplumber` and `python-docx`
- Chunk and embed text using SentenceTransformer (`all-MiniLM-L6-v2`)
- Store embeddings with FAISS for efficient semantic search
- Query using a lightweight LLM (`google/flan-t5-base`) running on CPU
- Fully free to run on Hugging Face Spaces (no paid APIs or GPU needed)

## How to Use

1. Upload a PDF or DOCX document.
2. Enter a question related to the document content.
3. Get an AI-generated answer based on semantic search and language modeling.

## Technical Details

- Streamlit for UI
- `pdfplumber` to extract text from PDF
- `python-docx` to extract text from DOCX
- SentenceTransformer for embeddings
- FAISS for vector similarity search
- `google/flan-t5-base` for CPU-based generative Q&A

## Deployment

This app is designed to run on free Hugging Face Spaces using Streamlit.

---

## How to Deploy on Hugging Face Spaces

1. Create a new Space on Hugging Face, select Streamlit as the SDK, and CPU hardware.
2. Clone your local project or directly upload your folder to the Space repo.
3. Make sure your folder contains:
   - `app.py`
   - `requirements.txt`
   - `README.md`
   - `src` folder with your code modules
4. The Space will automatically install dependencies and launch the app.

---

Feel free to open issues for bugs or feature requests.  
Developed by Harsha M Purohit.
