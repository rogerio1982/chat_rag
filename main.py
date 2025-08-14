"""
FastAPI Multimodal RAG API with Gemini
--------------------------------------
Endpoints:
POST /support        -> accepts text + optional image
POST /support/audio  -> accepts audio (.mp3/.wav) + optional image
"""

import os
import io
import uuid
import tempfile
import json
from typing import List, Optional

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from PIL import Image
import pytesseract
import PyPDF2
import whisper
from gtts import gTTS
from sentence_transformers import SentenceTransformer
import faiss
import re

# Gemini
import google.generativeai as genai

# Configure Gemini API key
genai.configure(api_key="AIzaSyDDAXLbSQBeC0JGNdCAafQ9Af0WPAIy0Yo")

# Directories
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
os.makedirs(STATIC_DIR, exist_ok=True)
KB_DIR = os.path.join(BASE_DIR, "knowledge_base")
os.makedirs(KB_DIR, exist_ok=True)
INDEX_DIR = os.path.join(BASE_DIR, "faiss_index")
os.makedirs(INDEX_DIR, exist_ok=True)

# Models
WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "small")
EMBEDDING_MODEL_NAME = os.environ.get("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
TOP_K = int(os.environ.get("RAG_TOP_K", "3"))

app = FastAPI(title="Multimodal RAG API - Gemini")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Globals
_whisper_model = None
_embeddings_model = None
_faiss_index = None
_documents = []
_texts = []

# --- Utility functions ---

def chunk_text(text, max_len=300):
    words = text.split()
    return [" ".join(words[i:i+max_len]) for i in range(0, len(words), max_len)]

def load_whisper_model():
    global _whisper_model
    if _whisper_model is None:
        _whisper_model = whisper.load_model(WHISPER_MODEL)
    return _whisper_model

def load_embeddings_model():
    global _embeddings_model
    if _embeddings_model is None:
        _embeddings_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _embeddings_model

def index_exists():
    return os.path.exists(os.path.join(INDEX_DIR, "index.faiss")) and os.path.exists(os.path.join(INDEX_DIR, "meta.json"))

def save_index(index, metas, texts):
    faiss.write_index(index, os.path.join(INDEX_DIR, "index.faiss"))
    data = {"metas": metas, "texts": texts}
    with open(os.path.join(INDEX_DIR, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)

def load_index():
    global _faiss_index, _documents, _texts
    idx_path = os.path.join(INDEX_DIR, "index.faiss")
    meta_path = os.path.join(INDEX_DIR, "meta.json")

    _faiss_index = faiss.read_index(idx_path)

    with open(meta_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        _documents = data.get("metas", [])
        _texts = data.get("texts", [])
    else:
        print("Error: meta.json is not in the expected dictionary format.")
        _documents = []
        _texts = []

    return _faiss_index

def build_index_from_kb():
    global _faiss_index, _documents, _texts
    emb = load_embeddings_model()
    texts, metas = [], []
    for root, _, files in os.walk(KB_DIR):
        for fn in files:
            path = os.path.join(root, fn)
            ext = fn.lower().split('.')[-1]
            text = ""
            try:
                if ext in ("txt", "md"):
                    with open(path, "r", encoding="utf-8", errors='ignore') as f:
                        text = f.read()
                elif ext == "pdf":
                    with open(path, "rb") as f:
                        reader = PyPDF2.PdfReader(f)
                        text = "\n".join([p.extract_text() or "" for p in reader.pages])
            except Exception as e:
                print(f"Error reading {path}: {e}")
                continue
            if text.strip():
                for idx, chunk in enumerate(chunk_text(text)):
                    texts.append(chunk)
                    metas.append({"source": fn, "path": path, "chunk_id": idx})
    if not texts:
        return None
    embeddings = emb.encode(texts, convert_to_numpy=True)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    save_index(index, metas, texts)
    _faiss_index = index
    _documents = metas
    _texts = texts
    return index

def retrieve(query: str, top_k=TOP_K):
    emb = load_embeddings_model()
    global _faiss_index, _documents, _texts
    if _faiss_index is None:
        if index_exists():
            load_index()
        else:
            build_index_from_kb()
    if _faiss_index is None:
        return []
    q_emb = emb.encode([query], convert_to_numpy=True)
    _, I = _faiss_index.search(q_emb, top_k)
    results = []
    for idx in I[0]:
        if idx < 0 or idx >= len(_documents):
            continue
        meta = _documents[idx]
        snippet = _texts[idx][:1000] if idx < len(_texts) else ""
        results.append({"source": meta['source'], "snippet": snippet})
    return results

def generate_answer(question: str, context_docs: List[dict]) -> str:
    """
    Generates a technical and friendly answer using Gemini LLM,
    based on support documents.
    """
    if not context_docs:
        return (
            "Hello! I could not find relevant information in my documents to answer your question. "
            "Please rephrase the question or provide more details."
        )

    context_text = "\n\n".join([f"[{d['source']}]: {d['snippet']}" for d in context_docs])

    prompt = (
        "You are a **technical support assistant**. Your job is to answer questions clearly, "
        "accurately, and friendly, as if helping a coworker or a client. "
        "Use the information provided in the context below to formulate your answer.\n\n"
        "**Instructions:**\n"
        "1. **Be concise:** Get straight to the point.\n"
        "2. **Maintain a professional and helpful tone:** Use accessible language, avoid slang and complex jargon unless necessary.\n"
        "3. **Cite sources when possible:** Mention sources if applicable (e.g., 'According to the manual, ...').\n"
        "4. **If information is insufficient:** Politely say the information was not found in the reference material. Do not invent answers.\n"
        "5. **Format the response:** Use lists, bold, or italics for readability, especially for step-by-step instructions.\n\n"
        f"**Context:**\n{context_text}\n\n**Question:** {question}\n**Answer:**"
    )

    model = genai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content(prompt)

    return response.text

def transcribe_audio(file_path: str) -> str:
    model = load_whisper_model()
    res = model.transcribe(file_path)
    return res.get('text', '').strip()

def ocr_image(image_path: str) -> str:
    try:
        img = Image.open(image_path)
        return pytesseract.image_to_string(img).strip()
    except Exception:
        return ""

def text_to_speech(text: str, out_path: str) -> str:
    tts = gTTS(text=text, lang='en')
    tts.save(out_path)
    return out_path

# --- Schemas ---
class SupportResponse(BaseModel):
    transcription: Optional[str]
    ocr_text: Optional[str]
    answer: str
    audio_url: Optional[str]
    source_documents: List[str]

# --- Endpoints ---

@app.post('/support', response_model=SupportResponse)
async def support_endpoint(
    text: Optional[str] = Form(None),
    image: Optional[UploadFile] = File(None)
):
    if not text and not image:
        raise HTTPException(status_code=400, detail="Provide 'text' or 'image'.")

    ocr_text = None
    if image:
        suffix = os.path.splitext(image.filename)[1].lower()
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        tmp.write(await image.read())
        tmp.close()
        ocr_text = ocr_image(tmp.name)
        os.unlink(tmp.name)

    query = text or ""
    if ocr_text:
        query += "\n" + ocr_text

    docs = retrieve(query)
    sources = [d['source'] for d in docs]
    answer = generate_answer(query, docs)

    mp3_name = f"response_{uuid.uuid4().hex}.mp3"
    mp3_path = os.path.join(STATIC_DIR, mp3_name)
    try:
        text_to_speech(answer, mp3_path)
        audio_url = f"/static/{mp3_name}"
    except Exception:
        audio_url = None

    return SupportResponse(transcription=None, ocr_text=ocr_text, answer=answer, audio_url=audio_url, source_documents=sources)

@app.post('/support/audio', response_model=SupportResponse)
async def support_audio_endpoint(
    audio: UploadFile = File(...),
    image: Optional[UploadFile] = File(None)
):
    suffix = os.path.splitext(audio.filename)[1].lower()
    tmp_audio = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp_audio.write(await audio.read())
    tmp_audio.close()
    transcription = transcribe_audio(tmp_audio.name)
    os.unlink(tmp_audio.name)

    ocr_text = None
    if image:
        img_suffix = os.path.splitext(image.filename)[1].lower()
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=img_suffix)
        tmp.write(await image.read())
        tmp.close()
        ocr_text = ocr_image(tmp.name)
        os.unlink(tmp.name)

    query = transcription or ""
    if ocr_text:
        query += "\n" + ocr_text

    docs = retrieve(query)
    sources = [d['source'] for d in docs]
    answer = generate_answer(transcription, docs)

    mp3_name = f"response_{uuid.uuid4().hex}.mp3"
    mp3_path = os.path.join(STATIC_DIR, mp3_name)
    try:
        text_to_speech(answer, mp3_path)
        audio_url = f"/static/{mp3_name}"
    except Exception:
        audio_url = None

    return SupportResponse(transcription=transcription, ocr_text=ocr_text, answer=answer, audio_url=audio_url, source_documents=sources)

@app.get('/health')
async def health_check():
    return {"status": "ok"}
